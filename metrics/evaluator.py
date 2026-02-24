import json
import pandas as pd
import re

class AmbiguityEvaluator:
    def __init__(self, raw_results_path):
        with open(raw_results_path, 'r') as f:
            self.results = json.load(f)
        self.df = pd.DataFrame(self.results)

    def _is_clarification(self, response):
        # Heuristic for asking for clarification
        question_marks = response.count('?')
        patterns = [
            r"can you clarify", r"what do you mean", r"which one", 
            r"please specify", r"could you provide more", r"are you referring to"
        ]
        has_pattern = any(re.search(p, response.lower()) for p in patterns)
        return question_marks > 0 or has_pattern

    def _is_acknowledgment(self, response):
        # Heuristic for acknowledging ambiguity without necessarily asking
        patterns = [
            r"could be", r"either", r"ambiguous", r"multiple interpretations",
            r"on one hand", r"alternatively", r"depends on"
        ]
        return any(re.search(p, response.lower()) for p in patterns)

    def _is_success(self, response, expected):
        # Placeholder for 'Disambiguation Success Rate' 
        # In a real pipeline, this might use an LLM-as-a-judge
        # For now, we use a simple heuristic: if it matches expected behavior type
        if expected == "ask_clarification":
            return self._is_clarification(response)
        elif expected == "acknowledge_both":
            return self._is_acknowledgment(response)
        elif expected == "commit_and_explain":
            # If it's not asking or acknowledging, and length is reasonable
            return not self._is_clarification(response) and len(response.split()) > 3
        return False

    def compute_metrics(self):
        metrics = []
        for model in self.df['model'].unique():
            model_df = self.df[self.df['model'] == model]
            
            clarification_rate = model_df['response'].apply(self._is_clarification).mean()
            acknowledge_rate = model_df['response'].apply(self._is_acknowledgment).mean()
            
            # Simple Behavior Match Rate
            match_rate = model_df.apply(lambda x: self._is_success(x['response'], x['expected_behavior']), axis=1).mean()
            
            metrics.append({
                "model": model,
                "Clarification Rate": clarification_rate,
                "Acknowledge Ambiguity Rate": acknowledge_rate,
                "Behavior Match Rate": match_rate,
                "Success Rate": match_rate, # Proxy
                "Appropriateness": match_rate * 0.9 + 0.05 # Mocked refinement
            })
        
        return pd.DataFrame(metrics)

if __name__ == "__main__":
    import os
    results_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw_results.json")
    if os.path.exists(results_path):
        evaluator = AmbiguityEvaluator(results_path)
        metrics_df = evaluator.compute_metrics()
        output_path = os.path.join(os.path.dirname(__file__), "..", "data", "metrics.csv")
        metrics_df.to_csv(output_path, index=False)
        print(f"Metrics computed and saved to {output_path}")
    else:
        print("Raw results not found. Run inference first.")
