import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy import stats
import json
import os

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

def run_statistical_tests(metrics_path, raw_results_path):
    # For statistical significance, we need the per-probe scores
    with open(raw_results_path, 'r') as f:
        raw_data = json.load(f)
    
    # We'll use the 'Match' status of each probe as the sample
    # (Simplified: 1 if success, 0 otherwise)
    from metrics.evaluator import AmbiguityEvaluator
    evaluator = AmbiguityEvaluator(raw_results_path)
    df = pd.DataFrame(raw_data)
    
    # Add success column
    df['success'] = df.apply(lambda x: evaluator._is_success(x['response'], x['expected_behavior']), axis=1).astype(int)
    
    models = df['model'].unique()
    comparisons = []
    
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            scores1 = df[df['model'] == m1]['success'].values
            scores2 = df[df['model'] == m2]['success'].values
            
            # Paired T-test
            t_stat, p_val_t = stats.ttest_rel(scores1, scores2)
            
            # Wilcoxon signed-rank test
            try:
                w_stat, p_val_w = stats.wilcoxon(scores1, scores2)
            except ValueError:
                # Occurs if all differences are zero
                w_stat, p_val_w = 0, 1.0
            
            # Cohen's d
            d = cohen_d(scores1, scores2)
            
            comparisons.append({
                "comparison": f"{m1} vs {m2}",
                "t_stat": t_stat,
                "p_value_t": p_val_t,
                "wilcoxon_stat": w_stat,
                "p_value_w": p_val_w,
                "cohen_d": d
            })
            
    return pd.DataFrame(comparisons)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_path = os.path.join(base_dir, "data", "raw_results.json")
    metrics_path = os.path.join(base_dir, "data", "metrics.csv")
    
    if os.path.exists(raw_path):
        stats_df = run_statistical_tests(metrics_path, raw_path)
        output_path = os.path.join(base_dir, "data", "stats_results.csv")
        stats_df.to_csv(output_path, index=False)
        print(f"Statistical tests completed and saved to {output_path}")
    else:
        print("Raw results not found.")
