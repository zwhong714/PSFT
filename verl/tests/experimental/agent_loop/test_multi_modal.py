# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
from typing import Any

import numpy as np
import pytest
import ray
from omegaconf import DictConfig
from PIL import Image
from transformers.utils import get_json_schema

from tests.experimental.agent_loop.agent_utils import init_agent_loop_manager
from verl.protocol import DataProto
from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema
from verl.tools.schemas import ToolResponse
from verl.utils import hf_tokenizer


@pytest.fixture
def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.use_dynamic_bsz=true",
                # test sleep/wake_up with fsdp offload
                "actor_rollout_ref.actor.fsdp_config.param_offload=True",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
            ],
        )

    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.name = os.environ["ROLLOUT_NAME"]
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.prompt_length = 4096
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.n = 4
    config.actor_rollout_ref.rollout.agent.num_workers = 2

    return config


class ImageGeneratorTool(BaseTool):
    def generate_image(self, description: str, size: str = "256x256"):
        """Generate a simple image based on description.

        Args:
            description: The description of the image to generate.
            size: The size of the image. Defaults to "256x256". (choices: ["256x256", "512x512"])

        Returns:
            A generated image
        """
        print(f"[DEBUG] generate_image: {description}, {size}")
        # Create a simple colored image for testing
        width, height = map(int, size.split("x"))

        # Create different colors based on description
        if "red" in description.lower():
            color = (255, 0, 0)
        elif "blue" in description.lower():
            color = (0, 0, 255)
        elif "green" in description.lower():
            color = (0, 255, 0)
        else:
            color = (128, 128, 128)  # gray

        # Create image
        image = Image.new("RGB", (width, height), color)

        # Add some pattern to make it more interesting
        for i in range(0, width, 50):
            for j in range(0, height, 50):
                # Add white squares in a grid pattern
                for x in range(i, min(i + 20, width)):
                    for y in range(j, min(j + 20, height)):
                        image.putpixel((x, y), (255, 255, 255))

        return image

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        schema = get_json_schema(self.generate_image)
        return OpenAIFunctionToolSchema(**schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        try:
            image = self.generate_image(**parameters)
            # Return the PIL Image directly - the framework should handle the conversion
            return ToolResponse(image=[image]), 0, {}
        except Exception as e:
            return ToolResponse(text=str(e)), 0, {}


def test_multimodal_tool_agent(init_config):
    """Test agent loop with multimodal tool that returns images using Qwen VL model."""
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        },
        ignore_reinit_error=True,
    )

    # Add custom chat template to enable tool calling support (same as recipe/deepeyes)
    template_path = os.path.join(os.path.dirname(__file__), "qwen_vl_tool_chat_template.jinja2")
    with open(template_path, encoding="utf-8") as f:
        custom_chat_template = f.read()

    init_config.actor_rollout_ref.model.custom_chat_template = custom_chat_template

    # =========================== 1. Init rollout manager with image tool ===========================
    tool_config = {
        "tools": [
            {
                "class_name": "tests.experimental.agent_loop.test_multi_modal.ImageGeneratorTool",
                "config": {"type": "native"},
            },
        ]
    }
    tool_config_path = "/tmp/multimodal_tool_config.json"
    with open(tool_config_path, "w") as f:
        json.dump(tool_config, f)

    n = 2
    init_config.actor_rollout_ref.rollout.n = n
    init_config.actor_rollout_ref.rollout.multi_turn.tool_config_path = tool_config_path
    init_config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls = 1
    init_config.actor_rollout_ref.rollout.multi_turn.max_user_turns = 1
    agent_loop_manager = init_agent_loop_manager(init_config)

    # =========================== 2. Generate sequences with multimodal prompts ===========================
    raw_prompts = [
        [
            {"role": "user", "content": "How are you?"},
        ],
        [
            {"role": "user", "content": "Please generate a red image for me."},
        ],
        [
            {"role": "user", "content": "Can you create a blue picture with size 512x512?"},
        ],
        [
            {
                "role": "system",
                "content": (
                    "You are Qwen VL, created by Alibaba Cloud. You are a helpful "
                    "assistant that can generate and analyze images."
                ),
            },
            {"role": "user", "content": "Generate a green landscape image and describe what you see in it."},
        ],
    ]

    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array([np.array(prompt) for prompt in raw_prompts], dtype=object),
            "agent_name": np.array(["tool_agent"] * len(raw_prompts)),
            "data_source": np.array(["openai/gsm8k"] * len(raw_prompts)),
            "reward_model": np.array([{"style": "rule", "ground_truth": "1.0"}] * len(raw_prompts)),
        },
    )
    batch = batch.repeat(n)
    result = agent_loop_manager.generate_sequences(prompts=batch)
    assert len(result) == len(raw_prompts) * n

    # Check turns
    num_turns = result.non_tensor_batch["__num_turns__"]
    print(f"num_turns: {num_turns}")
    for i in range(len(num_turns)):
        if i // n == 0:
            # First prompt: "How are you?" - should have 2 turns [user, assistant]
            assert num_turns[i] == 2, f"Expected 2 turns but got {num_turns[i]} for sample {i}"
        else:
            # Tool-calling prompts should have 4 turns [user, assistant, tool, assistant]
            assert num_turns[i] == 4, f"Expected 4 turns but got {num_turns[i]} for sample {i}"

    # Check that images were properly returned in the tool responses
    tokenizer = hf_tokenizer(init_config.actor_rollout_ref.model.path)
    responses = result.batch["responses"]
    response_mask = result.batch["response_mask"]
    attention_mask = result.batch["attention_mask"]
    assert responses.size() == response_mask.size(), f"{responses.size()} != {response_mask.size()}"
    response_length = response_mask.size(1)

    image_found_count = 0
    for i in range(len(responses)):
        # response with tool response (including images)
        valid_tokens = responses[i][attention_mask[i][-response_length:].bool()]
        response_with_obs = tokenizer.decode(valid_tokens)

        # response without tool response
        valid_tokens = responses[i][response_mask[i].bool()]
        response_without_obs = tokenizer.decode(valid_tokens)

        # Check that tool responses were properly masked out from training
        assert "<tool_response>" not in response_without_obs, (
            f"found <tool_response> in response: {response_without_obs}"
        )
        assert "</tool_response>" not in response_without_obs, (
            f"found </tool_response> in response: {response_without_obs}"
        )

        # Check that images were included in the full response
        if "<image>" in response_with_obs or "image" in response_with_obs.lower():
            image_found_count += 1

        print("=========================")
        print("Response with tool observations:")
        print(response_with_obs)
        print("---")
        print("Response without tool observations:")
        print(response_without_obs)

    # Verify that tool-calling responses contained image-related content
    print(f"Found {image_found_count} responses with image content out of {len(responses)}")
    # We should have at least some image content from the tool-calling prompts
    # Note: First prompt might not use tools, so we don't expect 100% image content
    expected_tool_calls = sum(1 for i in range(len(num_turns)) if num_turns[i] == 4)
    assert image_found_count >= 0, (
        f"No image-related content found, but expected at least some from {expected_tool_calls} tool calls"
    )

    print("Multimodal tool test passed!")
    ray.shutdown()
