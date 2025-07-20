## LLM Finetuning Framework

Important Resource(LLM Leaderboard), Framework to Goal Analysis, Finetuning enterprise level expectation, Important research paper.

### LLM Finetuning Framework

1.  Hugging Face for LLM Fine-Tuning
    *   Huge Model Hub (100,000+ Models)
    *   Transformers Library = Industry Standard
    *   Fine-Tuning Made Easy
    *   PEFT + LoRA for Efficient Finetuning
    *   Datasets Library
    *   Trainer + Accelerate = Fast & Distributed
    *   Hugging Face Hub
    *   Evaluation, Tokenizers, & Inference Support
2.  DeepSpeed
    *   Memory Optimization (ZeRO)
    *   Faster Training (CPU Offloading, Kernel Fusion)
    *   Gradient Accumulation + Mixed Precision
    *   Model Parallelism + Pipeline Parallelism
    *   Checkpointing + Resume + Logging
3.  LLaMA Factory
    *   LoRA / QLoRA Fine-Tuning
    *   Minimal Setup, CLI-Friendly
    *   Supports 100+ Hugging Face LLMs
    *   Multi-GPU, DeepSpeed, FSDP(Fully Sharded Data Parallel), Flash Attention (super-optimized attention algorithm )
    *   Chat Templates + SFT Format Support
    *   Integrated Evaluation + MMLU/TruthfulQA
    *   Export to GGUF / GPTQ / HF
4.  Unsloth
    *   Fastest LoRA Fine-Tuning
    *   Low VRAM Usage - Train LLM on Colab T4!
    *   One-Line LLM Loading (with LoRA-ready model)
    *   Integrated PEFT + Hugging Face Compatibility
    *   Supports Multiple Model and Multimodal
    *   Torch Compile + Flash Attention
5.  Axolotol

### Syllabus of finetuning

1.  Video 01: Introduction to Fine-Tuning in AI
    *   What is Model Training (ML, DL, CV, NLP, Generative AI)
    *   Transfer Learning (Live Demo with CNN)
    *   What is Pretraining, Finetuning & Why it matters
    *   Pros & Cons of Finetuning
    *   Overview of Top LLM Finetuning Frameworks: HF TRL, Unsloth, Llama Factory, Axolotl
    *   Important Research Papers
    *   Tips Before You Fine-Tune
2.  Video 02: Finetuning vs RAG vs AI Agents - What to Use When?
    *   Definitions: Finetuning, Transfer Learning, RAG, Agents
    *   Comparison Table
    *   Use Cases & When to use What:
    *   Industry Examples
3.  Video 03: Finetuning in Deep Learning
    *   CNN Example End-to-End (Image Classification)
    *   How feature extraction + finetuning works
4.  Video 04: Why Finetuning is not possible in RNN/LSTM?
    *   Limitations of older architectures (RNN, LSTM)
    *   Compare with Transformer-based models
5.  Video 05: Hugging Face vs LangChain | Full HF Tutorial
    *   HF Installation & Environment Setup
    *   Using HF APIs vs Offline Downloads
6.  Video 06: Finetuning Classical Language Models
    *   BERT Fine-Tuning (Text Classification / QA)
    *   TS Fine-Tuning (Text2Text Tasks)
7.  Video 07: Knowledge Distillation in LLMs
    *   What is Knowledge Distillation?
    *   Example: BERT -> Distil BERT
    *   Combine with Fine-Tuning for better efficiency
8.  Video 08: LLM Quantization Explained
    *   What is Quantization?
    *   Weight Quantization Techniques
    *   GGUF, GGML, GPTQ, AWQ, INTA/INT8 demos
    *   Why quantization matters
9.  Video 09: Finetuning Large Language Models (Llama, Mistral, Gemma, Phi-3)
    *   One-command fine-tuning demo
    *   PEFT Techniques: LoRA, QLoRA, DoRA, ReLoRa, deFTy
    *   Structured output fine-tuning (OpenAssistant-like)
    *   Chat-based fine-tuning (OpenAssistant-like)
    *   Full finetuning vs parameter-efficient finetuning
    *   Dataset: Wiki dataset, DPO/D4
    *   Tools: Axolotl, Apple MLX, Unsloth
    *   Post-Finetune Deployment (Ollama Integration)
10. Video 10: API-Based Model Finetuning (GPT-4o, Gemini etc)
    *   OpenAI Finetuning Walkthrough
    *   Distillation as IR for All models
    *   Gemini Finetune Insights
    *   GPT-4o use case finetuning
11. Video 11: Best Frameworks for LLM Fine-Tuning
    *   Hands-on: LlamaFactory, Unsloth
    *   Minimal Code, Max Performance
    *   Compare speed, memory, and flexibility
12. Video 12: Finetuning Vision-Language Models
    *   What are VLMs? (ViT, Florence, OpeniV2, LaGemini)
    *   Finetuning VLMs with LlamaFactory
    *   Upload Adapters & Models to Hugging Face Hub
13. Video 13: Reinforcement Learning with Human Feedback (RLHF)
    *   PPO vs DPO vs Direct Preference Optimization
    *   Real Examples and when to use what
    *   How RLHF fits into finetuning pipeline
14. Video 14: Embedding Finetuning Deep Dive
    *   Embedding vs Fine-Tuning: Conceptual & Practical
    *   SFT vs USFT (Supervised Fine-Tuning variants)
    *   Finetuning for Retrieval & Semantic Search
    *   When to finetune embeddings vs models
