import os
import argparse
import datasets

from datasets import load_dataset


train_data = load_dataset("HuggingFaceH4/aime_2024", split='train')


def process_fn_train(example, idx):

    question = example.pop("problem")
    
    solution =  example.pop("answer")
    idx = example.pop("id")
        
    data = {
        "data_source": "lighteval/MATH",
        "prompt": [
            {
                'role': 'system', 'content': r"Please reason step by step, and put your final answer within \boxed{}."
            },
            {
                "role": "user",
                "content": question,
            }
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {
            "index": idx,
            "raw_problem": question_raw,
        },
    }
    return data


train_dataset = train_data.map(function=process_fn_train, with_indices=True)
train_dataset.to_parquet(os.path.join('./', 'aime24.parquet'))
print(f"Train: {len(train_dataset)}")

