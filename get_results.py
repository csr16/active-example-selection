import json
from pathlib import Path

import pandas as pd

def summarize_acc(base_path, methods, datasets):
    """
    Scan all subdirectories of `base_path` of the form method_dataset_seed,
    read the 'acc' field from each eval.jsonl, and return a DataFrame
    with mean and std of acc for each method–dataset combination.
    """
    records = []
    base = Path(base_path)

    for subdir in base.iterdir():
        if not subdir.is_dir():
            continue

        parts = subdir.name.split('_')
        # if len(parts) != 3:
        #     continue

        method, dataset, seed = parts[0], '_'.join(parts[1:-1]), parts[-1]
        if method not in methods or dataset not in datasets:
            continue

        eval_file = subdir / 'eval.jsonl'
        if not eval_file.exists():
            continue

        # read all JSON lines and collect 'acc'
        with eval_file.open('r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                acc = data.get('acc')
                if acc is not None:
                    records.append({
                        'method': method,
                        'dataset': dataset,
                        'seed': int(seed),
                        'acc': float(acc)
                    })

    df = pd.DataFrame(records)
    # compute mean and std
    stats = (
        df
        .groupby(['method', 'dataset'])['acc']
        .agg(mean='mean', std='std')
        .reset_index()
    )
    return stats

if __name__ == '__main__':
    # === user parameters ===
    base_path = './outputs/Qwen_Qwen2_5-3B'
    methods   = ['random', 'max-entropy', 'best-of-k', 'global-entropy-ordering']  # <-- fill in your methods
    datasets  = ['agnews', 'sst-2', 'winowhy', 'epistemic_reasoning']  # <-- fill in your datasets

    # run
    summary_df = summarize_acc(base_path, methods, datasets)
    print(summary_df.to_string(index=False))