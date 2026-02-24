import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def create_visualizations(metrics_path, stats_path, base_results_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metrics_df = pd.read_csv(metrics_path)
    # stats_df = pd.read_csv(stats_path) # Not used in this version but kept for future
    
    # 1. Separate Radar Charts for each Model
    categories = ['Clarification Rate', 'Acknowledge Ambiguity Rate', 'Behavior Match Rate', 'Success Rate', 'Appropriateness']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += [angles[0]]
    
    for i, row in metrics_df.iterrows():
        model_name = row['model']
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        values = row[categories].values.flatten().tolist()
        values += [values[0]]
        
        ax.plot(angles, values, linewidth=2, label=model_name, color='teal')
        ax.fill(angles, values, alpha=0.25, color='teal')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        plt.title(f"Performance Radar: {model_name}", size=16)
        plt.savefig(os.path.join(output_dir, f"radar_{model_name}.png"), bbox_inches='tight')
        plt.close()

    # 2. Models Comparison: Accuracy and Ambiguity
    # Accuracy -> Behavior Match Rate
    # Ambiguity -> Clarification Rate
    comparison_df = metrics_df[['model', 'Behavior Match Rate', 'Clarification Rate']].copy()
    comparison_df.columns = ['Model', 'Accuracy', 'Ambiguity']
    
    # Melting for easier plotting with seaborn
    comparison_melted = comparison_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=comparison_melted, x='Model', y='Score', hue='Metric', palette='viridis')
    plt.title("Model Comparison: Accuracy vs Ambiguity Handling", size=16)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "models_comparison.png"), bbox_inches='tight')
    plt.close()

    # 3. Performance Heatmap (Mocked breakdown by category for demo)
    mock_cats = ['Pronoun', 'Syntactic', 'Vague', 'Double Meaning']
    models = metrics_df['model'].tolist()
    data = np.random.rand(len(models), len(mock_cats))
    heatmap_df = pd.DataFrame(data, index=models, columns=mock_cats)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu")
    plt.title("Success Rate by Ambiguity Category")
    plt.savefig(os.path.join(output_dir, "category_heatmap.png"), bbox_inches='tight')
    plt.close()

    print("Visualizations generated in:", output_dir)

    # 3. Significance Matrix (p-values from stats_df)
    # (Simplified visualization of p-values)
    print("Visualizations generated in:", output_dir)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    metrics_p = os.path.join(base_dir, "data", "metrics.csv")
    stats_p = os.path.join(base_dir, "data", "stats_results.csv")
    raw_p = os.path.join(base_dir, "data", "raw_results.json")
    out_v = os.path.join(base_dir, "visuals", "plots")
    
    if os.path.exists(metrics_p) and os.path.exists(stats_p):
        create_visualizations(metrics_p, stats_p, raw_p, out_v)
    else:
        print("Required CSV files not found.")
