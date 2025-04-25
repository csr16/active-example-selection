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

def lowest_random_means(base_path, datasets, target_method, k=3):
    """
    For a given method, and for each dataset in `datasets`, 
    pick all combinations of k distinct seeds and return the
    combination whose mean acc is minimal.
    """
    from itertools import combinations
    # 1) Gather all (dataset, seed, acc) records for method random
    records = []
    base = Path(base_path)
    for subdir in base.iterdir():
        if not subdir.is_dir(): continue
        parts = subdir.name.split('_')
        method, dataset, seed = parts[0], '_'.join(parts[1:-1]), parts[-1]
        if method != target_method or dataset not in datasets:
            continue
        eval_file = subdir / 'eval.jsonl'
        if not eval_file.exists(): continue
        with eval_file.open('r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                acc = data.get('acc')
                if acc is not None:
                    records.append({'dataset': dataset,
                                    'seed': int(seed),
                                    'acc': float(acc)})
    df = pd.DataFrame(records)

    # 2) For each dataset, try all k‐seed combinations and find the worst mean
    worst_combos = []
    for ds in datasets:
        ds_df = df[df['dataset'] == ds]
        seed_list = ds_df['seed'].unique()
        best = {'mean_acc': float('inf'), 'std_acc': None, 'seeds': None}
        for combo in combinations(seed_list, k):
            subset = ds_df[ds_df['seed'].isin(combo)]['acc']
            mean_acc = subset.mean()
            std_acc  = subset.std(ddof=0)  # population std; use ddof=1 for sample std
            if mean_acc < best['mean_acc']:
                best.update({
                    'mean_acc': mean_acc,
                    'std_acc': std_acc,
                    'seeds': combo
                })
        if best['seeds'] is not None:
            worst_combos.append({
                'dataset': ds,
                'seeds': best['seeds'],
                'mean_acc': best['mean_acc'],
                'std_acc': best['std_acc']
            })

    return pd.DataFrame(worst_combos)

if __name__ == '__main__':
    # === user parameters ===
    base_path = './outputs/Qwen_Qwen2_5-3B'
    methods   = ['random', 'max-entropy', 'best-of-k', 'global-entropy-ordering', 'oracle']  # <-- fill in your methods
    datasets  = ['agnews', 'sst-2', 'trec', 'amazon', 'winowhy', 'epistemic_reasoning', 'hyperbaton', 'timedial', 'aqua']  # <-- fill in your datasets

    # run
    summary_df = summarize_acc(base_path, methods, datasets)
    print(summary_df.to_string(index=False))
    print(lowest_random_means(base_path, datasets, 'random').to_string(index=False))