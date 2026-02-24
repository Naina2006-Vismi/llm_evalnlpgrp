import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import json
import os

class InferenceEngine:
    def __init__(self, model_type="gpt2", device="cpu"):
        self.model_type = model_type
        self.device = device
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"Loading {self.model_type}...")
        if self.model_type == "gpt2":
            model_id = "distilgpt2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        elif self.model_type == "flan-t5":
            model_id = "google/flan-t5-small"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
        elif self.model_type == "llama-3":
            # Hook for LLaMA-3-8B
            # In a real scenario, this would load the meta-llama/Meta-Llama-3-8B model
            # For this pipeline demo, we'll use a placeholder or a smaller causal model
            print("LLaMA-3-8B placeholder: loading distilgpt2 as proxy")
            model_id = "distilgpt2" 
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        
        if self.model:
            self.model.eval()

    def generate(self, prompt, max_new_tokens=50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            if "t5" in self.model_type:
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            else:
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=0.0
                )
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # For causal models, remove the prompt part
        if "t5" not in self.model_type:
            if decoded.startswith(prompt):
                decoded = decoded[len(prompt):].strip()
        return decoded.strip()

def run_evaluation(data_path, models=["gpt2", "flan-t5", "llama-3"]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(data_path, 'r') as f:
        probes = json.load(f)
    
    results = []
    for m_type in models:
        engine = InferenceEngine(model_type=m_type, device=device)
        for probe in probes:
            prompt = f"Context: {probe['prompt']}\nQuestion: {probe['question']}\nResponse:"
            output = engine.generate(prompt)
            results.append({
                "model": m_type,
                "probe_id": probe["id"],
                "category": probe["category"],
                "prompt": probe["prompt"],
                "question": probe["question"],
                "response": output,
                "expected_behavior": probe["expected_behavior"]
            })
    
    return results

if __name__ == "__main__":
    # Test run
    data_file = os.path.join(os.path.dirname(__file__), "..", "data", "probes.json")
    eval_results = run_evaluation(data_file)
    output_file = os.path.join(os.path.dirname(__file__), "..", "data", "raw_results.json")
    with open(output_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"Inference completed. Results saved to {output_file}")
