# LLM-Based Requirements Ambiguity Classifier

A work-in-progress research pipeline for classifying ambiguity in 
software requirements using open-source LLMs — built as part of an 
AI internship at a project management consultancy.

## What this does
Classifies natural language software requirements as ambiguous or 
unambiguous using locally-deployed open-source LLMs, without sending 
data to external APIs (GDPR/EU AI Act compliant).

## Why it exists
Ambiguous requirements are a leading cause of project failure. This 
tool helps project managers flag unclear requirements early using AI — 
running entirely on standard laptops with no GPU required.

## Models evaluated so far
| Model | Parameters | Best strategy | Accuracy |
|---|---|---|---|
| Mistral 7B | 7B | Few-shot | 56.8% |
| Phi-3.5 Mini | 3.8B | Zero-shot | ~51% |
| Qwen 2.5 | 7B | RAG | ~54% |
| Llama 3 | 8B | Few-shot | ~53% |
| EuroLLM | 1.7B | Zero-shot | ~38% |

*Evaluation ongoing — prompt engineering experiments in progress.*

## Prompting strategies compared
- **Zero-shot**: Direct classification with no examples
- **Few-shot**: 3–5 labelled examples in context
- **RAG**: Retrieved similar requirements as context

Key finding so far: prompt strategy outweighs model size as a 
performance factor. A well-prompted 3B model outperforms a poorly 
prompted 7B model consistently.

## Technical stack
- `llama.cpp` for CPU inference
- 4-bit GGUF quantisation for local deployment
- Python pipeline for prompt formatting, inference, and output parsing
- Mistral API for synthetic training data generation

## Deployment constraint
Designed to run on standard customer laptops (no GPU). Entire 
inference pipeline runs on-premise — no data leaves the machine.

## Status
🔄 Active — currently optimising prompting strategies to push 
accuracy beyond 60% on the held-out test set.

## Note on data
The industrial requirements dataset is proprietary. Experiments on 
a public requirements dataset (PROMISE NFR / ESA) coming soon.
