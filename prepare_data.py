import os
import argparse
import datasets

from datasets import load_dataset


train_data = load_dataset("Elliott/Openr1-Math-46k-8192", split='train')


def process_fn_train(example, idx):
    data = {
        "data_source": "lighteval/MATH",
        "prompt": [
            {
                "role": "system",
                "content": r"Please reason step by step, and put your final answer within \boxed{}."
            },
            {
                "role": "user",
                "content": example["prompt"][1]['content']
            },                
        ],
        "ability": "math",
        "reward_model": example['reward_model'],
        "extra_info": {
            'split': 'train',
            'index': idx,
        },
        "demonstration": example['target'][0]['content']
    }
    return data


train_dataset = train_data.map(function=process_fn_train, with_indices=True)
train_dataset.to_parquet(os.path.join('./', 'train_openr1.parquet'))
print(f"Train: {len(train_dataset)}")

