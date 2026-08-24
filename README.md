This is a scratch project for R&D for making contributions to the Obliteratus open source project. This is a shadow of a fork of the source for the purpose of user privacy and security. Automation and tracking are in place under `aipithicus-issues` for development work, issue tracking, and pull request management.

## Surgery experiment lanes

The repository includes one experiment contract that runs both on a small local
GPU and in a Lightning AI Studio. Surgery is performed on pinned Hugging Face
Safetensors checkpoints; optional GGUF conversion and llama.cpp A/B inference
happen afterward.

- Local miniature profile: `experiments/surgery/local-qwen25-0.5b.yaml`
- Lightning scale profile: `experiments/surgery/lightning-qwen25-7b.yaml`
- Operator guide: [`docs/surgery-experiment-bench.md`](docs/surgery-experiment-bench.md)

