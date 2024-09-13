# Align-Anything Examples

Align-Anything is capable of supporting multiple models for training on various multimodal datasets. To help users get started quickly, we provide some simple examples here based on the most common open-source models, demonstrating how to use common alignment algorithms to fine-tune them.

- [Llava-v1.5-7B](./llava.md) LLaVA is an open-source chatbot trained by fine-tuning Vicuna on GPT-generated `Text+Image -> Text` modality instruction-following data. It is an auto-regressive language model, based on the transformer architecture.
- [Meta-Llama-3-8B](./llama.md) Llama 3 is a language model on `Text+Image -> Text` modality that uses an optimized transformer architecture.
- [Diffuision](./diffusion.md) Diffusion series models have achieved good performance in tasks involving the generation of multi-modal outputs from text (`Text -> Image`, `Text -> Audio`, `Text -> Video`). We also provide the corresponding fine-tuning code in this example file.