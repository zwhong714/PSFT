# Evaluation Command
```
serve run eval.llm:build_app model=aaa/bbb/ccc tensor-parallel-size=1

# open another terminal
python eval/eval.py --temperature 0.7 --top_p 0.95 --max_tokens 32768 --model ccc --test_file eval/data/aime-2024.parquet
```