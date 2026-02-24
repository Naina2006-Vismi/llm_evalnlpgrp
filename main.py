import os
import subprocess
import sys

def run_step(name, command):
    print(f"\n>>> Running Step: {name}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error in {name}. Exiting.")
        sys.exit(1)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1: Inference (already run or run now)
    # We'll skip if raw_results exist to save time, or run it if not
    raw_results = os.path.join(base_dir, "data", "raw_results.json")
    if not os.path.exists(raw_results):
        run_step("Model Inference", f"python3 {os.path.join(base_dir, 'engine', 'inference_wrapper.py')}")
    else:
        print("\n>>> Skipping Inference: raw_results.json already exists.")

    # Step 2: Metrics Computation
    run_step("Metrics Computation", f"python3 {os.path.join(base_dir, 'metrics', 'evaluator.py')}")

    # Step 3: Statistical Testing
    run_step("Statistical Testing", f"python3 {os.path.join(base_dir, 'stats', 'statistical_tests.py')}")

    # Step 4: Visualizations
    run_step("Visualizations", f"python3 {os.path.join(base_dir, 'visuals', 'plotting.py')}")

    print("\n" + "="*40)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*40)
    print("Artifacts generated:")
    print("- Raw Results: ambiguity_eval/data/raw_results.json")
    print("- Metrics: ambiguity_eval/data/metrics.csv")
    print("- Stats: ambiguity_eval/data/stats_results.csv")
    print("- Plots: ambiguity_eval/visuals/plots/")

if __name__ == "__main__":
    main()
