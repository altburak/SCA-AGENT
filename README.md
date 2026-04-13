# SCA: Stratified Cognitive Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

SCA (Stratified Cognitive Agent) is an advanced, self-calibrating AI agent architecture designed to solve complex tasks while actively evaluating its own confidence, grounding its claims in reality, and learning from past episodes.

The system leverages the Groq API and Llama 3.3 70B. Unlike standard LLM agents that blindly generate text, SCA employs a **Lean Adversarial Consensus** mechanism and a **Cross-Episode Distillation** pipeline to ensure high factual accuracy and metacognition ("knowing what it knows").

---

## Architecture Modules

The agent is built upon four core cognitive modules:

### PSM — Provenance-Stratified Memory

Tags all context blocks with their origin (`USER`, `EXTERNAL_TOOL`, `SELF_GENERATED`). This prevents the LLM from treating its own past hallucinations as ground truth.

### CSM — Confidence Scoring Module

Assigns a `0.0` to `1.0` confidence score to every generation using self-consistency, verifier models, and provenance penalties. Calibrates raw scores into actual accuracy probabilities using **Isotonic Regression** via `scikit-learn`.

### AOGL — Action-Outcome Grounding Loop

Proposes and executes real-world actions (reading files, executing sandboxed code, web scraping) to verify uncertain claims before finalizing an answer.

### CED — Cross-Episode Distillation

Extracts semantic insights from completed sessions, including bias patterns, successful strategies, failure modes, and domain knowledge. Uses `sentence-transformers` and automatically augments the system prompt for future episodes.

---

## Key Features

- **Lean Adversarial Consensus (v3.1):** Uses 3 orthogonal roles (Optimist, Evidence Literal, Minimal) in parallel to detect hallucinations via semantic disagreement.
- **Sandboxed Tool Execution:** Securely executes Python code and reads local files with SSRF protection and path traversal blocks.
- **Self-Calibrating:** Automatically tunes its confidence scores over time based on actual task success rates.
- **Automated Rate Limit Handling:** Manages multi-key API rotation seamlessly for high-throughput benchmarking.

---

## Quick Start

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/altburak/SCA-AGENT
cd SCA-Agent
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Set up environment variables:**

Create a `.env` file in the root directory and add your Groq API key:
```
GROQ_API_KEY=gsk_your_api_key_here
```

### Running Demos

```bash
python examples/psm_demo.py   # Memory management
python examples/csm_demo.py   # Confidence scoring
python examples/aogl_demo.py  # Grounding & tool execution
python examples/ced_demo.py   # Cross-episode learning
```

---

## Testing & Benchmarks

Run the test suite:
```bash
pytest tests/ -v
```

Run adversarial consensus benchmark:
```bash
python benchmarks/runner/run_mini_benchmark_v31.py
```