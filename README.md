# LLM Ambiguity Evaluation (`ambiguity_eval`)

A comprehensive pipeline for evaluating the behavior of Large Language Models (LLMs) when presented with ambiguous prompts and instructions.

## Overview

This project provides an automated evaluation framework to test how different LLMs handle various types of linguistic and functional ambiguities. It assesses whether models acknowledge ambiguity, ask for clarification, or wrongly commit to a single interpretation.

## Supported Ambiguity Categories

The dataset (`data/probes.json`) contains test probes across four main categories of ambiguity:
- **Pronoun Ambiguity**: e.g., Winograd schemas where a pronoun could refer to multiple entities.
- **Syntactic Ambiguity**: e.g., Prepositional Phrase (PP) attachment ambiguities.
- **Vague Instructions**: Prompts lacking necessary constraints or context (e.g., missing target language for translation).
- **Double Meaning**: Polysemy or Zeugma, where words have multiple distinct senses.

## Evaluated LLM Behaviors

The pipeline scores models based on their expected response behaviors for specific ambiguous prompts:
- `acknowledge_both`: The model acknowledges multiple interpretations of the prompt.
- `ask_clarification`: The model asks the user to clarify the underspecified instruction.
- `commit_and_explain`: The model commits to the most likely interpretation and explains its reasoning.

## Models Evaluated

The evaluation engine (`engine/inference_wrapper.py`) supports testing across different architectures:
- **GPT-2** (`distilgpt2`) - Causal LM
- **FLAN-T5-small** (`google/flan-t5-small`) - Seq2Seq LM
- **LLaMA-3** (Architecture Hook available)

## Pipeline Execution

The entire evaluation pipeline is orchestrated by `main.py`.

### Run the Pipeline

```bash
python3 main.py
```

### Pipeline Steps

When you run `main.py`, it executes the following steps automatically:

1. **Model Inference** (`engine/inference_wrapper.py`)
   - Feeds the probes to the LLMs and generates responses.
   - Output: `data/raw_results.json`

2. **Metrics Computation** (`metrics/evaluator.py`)
   - Computes evaluation metrics based on the models' responses.
   - Output: `data/metrics.csv`

3. **Statistical Testing** (`stats/statistical_tests.py`)
   - Performs statistical significance testing on the results.
   - Output: `data/stats_results.csv`

4. **Visualizations** (`visuals/plotting.py`)
   - Generates performance visualizations and category heatmaps.
   - Output: `visuals/plots/`

## Results and Artifacts

Upon completion, all generated artifacts are placed in the respective `data/` and `visuals/plots/` directories. These include detailed JSON responses, CSV stat summaries, and PNG radar/heatmap plots detailing the models' capabilities in handling ambiguity.