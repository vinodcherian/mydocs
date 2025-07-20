# Chapter 9: Fine-Tuning Large Language Models (Llama, Mistral, Gemma, Phi-3)

We've built a strong foundation. Now, it's time to apply these concepts to fine-tune some of the most popular and powerful open-source Large Language Models (LLMs) available today: Llama, Mistral, Gemma, and Phi-3. These models offer incredible capabilities, and fine-tuning allows you to tailor them to your specific needs.

**9.1 Introduction to the Models**

*   **Llama (by Meta AI):** A family of LLMs (Llama, Llama 2, Llama 3) released with open weights (for research and commercial use, with some restrictions). Known for strong performance and a vibrant open-source community.
*   **Mistral (by Mistral AI):** A series of high-performing open-weight models (e.g., Mistral 7B, Mixtral 8x7B). Mistral 7B, in particular, gained popularity for its excellent performance despite its relatively small size. Mixtral uses a "Mixture of Experts" (MoE) architecture for efficiency.
*   **Gemma (by Google):** Open models from Google, based on the same research and technology used to create their Gemini models. Available in different sizes (e.g., Gemma 2B, Gemma 7B).
*   **Phi-3 (by Microsoft):** A family of smaller, yet surprisingly capable, LLMs designed for efficiency and strong reasoning abilities, even at smaller parameter counts (e.g., Phi-3-mini).

`[Image: Logos of Llama, Mistral, Gemma, and Phi-3, perhaps with their respective creators noted.]`

**9.2 One-Command Fine-Tuning Demo (Conceptual)**

Many modern fine-tuning tools and frameworks (like Unsloth or LlamaFactory, which we'll discuss more in Chapter 11) aim to simplify the fine-tuning process dramatically. While the exact command varies, the idea is to make it as easy as possible.

**Conceptual "One-Command" Idea:**

```bash
# This is a highly simplified, conceptual example. Actual commands vary.
# Using a hypothetical tool:
super_finetuner --model_name "meta-llama/Llama-2-7b-hf" \
                --dataset_path "./my_instruction_dataset.jsonl" \
                --output_dir "./my_finetuned_llama" \
                --lora_rank 8 \
                --num_epochs 3 \
                --learning_rate 2e-4
```

This single (or few) command(s) would ideally handle:
*   Downloading the base model.
*   Loading and processing your dataset.
*   Setting up the training configuration (including PEFT methods like LoRA).
*   Running the fine-tuning loop.
*   Saving the fine-tuned model adapter or full model.

We'll see more concrete examples when we discuss specific tools.

**9.3 PEFT (Parameter-Efficient Fine-Tuning) Techniques**

Fine-tuning all the weights of a massive LLM (like Llama 70B) is still very computationally expensive and memory-intensive. Parameter-Efficient Fine-Tuning (PEFT) methods address this by only updating a small subset of the model's parameters, or by adding a small number of new, trainable parameters.

*   **Why PEFT?**
    *   **Drastically Reduces Compute & Memory:** Makes fine-tuning accessible on consumer GPUs.
    *   **Faster Training:** Fewer parameters to update.
    *   **Smaller Model Checkpoints:** You often only save the small set of changed/added parameters (adapters), not the entire model.
    *   **Reduces Catastrophic Forgetting:** By keeping most of the original model weights frozen.

**Popular PEFT Techniques:**

*   **LoRA (Low-Rank Adaptation):**
    *   **Concept:** Instead of updating the large weight matrices (W) of the Transformer layers directly, LoRA freezes W and injects two smaller, trainable "low-rank" matrices (A and B) next to it.
    *   During fine-tuning, only A and B are updated. The change to the original weights is represented by their product (B * A).
    *   `[Diagram: A Transformer layer's weight matrix W (frozen). Next to it, show two smaller matrices A and B (trainable) being multiplied and added to the output of W. W_new = W + B*A]`
    *   The `rank` (r) of A and B is a hyperparameter. A smaller rank means fewer trainable parameters.
    *   **Benefit:** Very effective, significantly reduces trainable parameters.

*   **QLoRA (Quantized Low-Rank Adaptation):**
    *   **Concept:** Combines LoRA with quantization.
    *   The base model's weights are quantized to a very low precision (e.g., 4-bit using NF4 - NormalFloat4 datatype) to save memory.
    *   LoRA adapters (A and B matrices) are then added and trained in a higher precision (e.g., FP16 or BF16).
    *   During the forward pass, the 4-bit base model weights are de-quantized on the fly to the computation precision, combined with the LoRA adapter outputs, and then the activations are processed.
    *   **Benefit:** Allows fine-tuning even larger models on single consumer GPUs by drastically reducing the memory footprint of the base model.

*   **DoRA (Weight-Decomposed Low-Rank Adaptation):**
    *   **Concept:** An improvement over LoRA. DoRA decomposes a pre-trained weight matrix into two components: magnitude (a vector) and direction (a matrix). It then applies LoRA to the directional component.
    *   Aims to improve the learning capacity and stability of LoRA, especially when the LoRA rank is small.

*   **ReFTy (Representation Finetuning, pronounced "Refty"):**
    *   **Concept:** A newer PEFT method that doesn't modify existing weights or add adapters directly to weight matrices. Instead, it focuses on fine-tuning a small portion of the model's *representations* (activations) at specific layers.
    *   It learns to modify a small, fixed number of hidden representations by applying a learned linear transformation (the ReFT intervention) to a subset of token positions at specific layers.
    *   **Claimed Benefit:** Can be even more parameter-efficient than LoRA for some tasks while achieving strong performance.

Hugging Face's `peft` library provides easy-to-use implementations for many of these techniques.

**9.4 Structured Output Fine-Tuning (e.g., QLoRA for JSON output)**

Sometimes, you need your LLM to generate output in a specific structured format, like JSON.
*   **Challenge:** LLMs are naturally text generators and might not always adhere strictly to complex structures.
*   **Approach:**
    1.  **Dataset Preparation:** Your fine-tuning dataset should consist of examples where the input is a prompt, and the desired output is a valid JSON string.
        ```json
        // Example fine-tuning data entry
        {
            "instruction": "Extract the name and email from the following text: John Doe (john.doe@email.com) is our new client.",
            "output": "{\"name\": \"John Doe\", \"email\": \"john.doe@email.com\"}"
        }
        ```
    2.  **Fine-Tuning (e.g., with QLoRA):** Fine-tune the LLM (e.g., Llama using QLoRA for efficiency) on this dataset. The model learns to generate text that looks like JSON.
    3.  **Prompting Techniques:** During inference, carefully craft your prompts to guide the model towards JSON output (e.g., "Extract the information as a JSON object with keys 'name' and 'email'.").
    4.  **Output Parsing & Validation:** After generation, you'll likely need to parse the output string to ensure it's valid JSON and extract the data.

**9.5 Chat-based Fine-Tuning (OpenAssistant-like)**

To create a conversational AI or chatbot:
*   **Dataset:** Requires datasets of multi-turn conversations. These datasets often follow a specific format, like the one used by OpenAssistant, where turns are marked with roles (e.g., `<|prompter|>`, `<|assistant|>`) and special end-of-turn tokens.
    `[Example:
    <|prompter|>Hello, how can you help me today?</s>
    <|assistant|>I can help you with a variety of tasks. What are you looking for?</s>
    <|prompter|>I need to book a flight.</s>
    <|assistant|>Sure, where would you like to fly from and to? ... </s>]`
*   **Fine-Tuning:** The LLM is fine-tuned on these conversational logs to learn how to engage in dialogue, maintain context, and respond appropriately as an assistant.
*   **Goal:** To make the model better at following instructions within a conversational context and generating helpful, coherent responses over multiple turns.

**9.6 Full Fine-Tuning vs. Parameter-Efficient Fine-Tuning**

| Aspect               | Full Fine-Tuning                               | PEFT (e.g., LoRA, QLoRA)                          |
| :------------------- | :--------------------------------------------- | :------------------------------------------------- |
| **Parameters Trained**| All (or most) model parameters                 | A small subset of parameters (e.g., 0.1% - 5%)      |
| **Compute/Memory**   | Very high                                      | Significantly lower                                |
| **Training Time**    | Longer                                         | Shorter                                            |
| **Checkpoint Size**  | Large (entire model weights)                   | Small (only adapter weights, if applicable)         |
| **Performance**      | Potentially the best, but risk of overfitting  | Often very close to full fine-tuning, more robust |
| **Catastrophic Forget**| Higher risk                                    | Lower risk                                         |
| **Use Case**         | When maximum performance is critical and resources are abundant | Most common approach for practical fine-tuning    |

**9.7 Dataset Prep: Wiki Dataset, FinWeb**

The quality of your fine-tuning dataset is crucial.
*   **General Format:** Often, instruction-following datasets are used, where each example has an "instruction" (prompt), an optional "input" (further context), and an "output" (desired response). JSONL (JSON Lines) is a common format.
*   **Wiki Dataset:** You could create a dataset from Wikipedia for tasks like question answering or summarization.
*   **FinWeb:** A dataset specifically for financial NLP tasks. It contains financial news, reports, and other text relevant to the finance domain. Using such domain-specific datasets is key for fine-tuning models for specialized applications.
    `[Image: A snippet of a JSONL dataset entry for instruction fine-tuning.]`
*   **Data Cleaning & Formatting:** This is a critical step. Ensure your data is clean, consistent, and formatted correctly for the fine-tuning script.

**9.8 Tools: Axolotl, Apple MLX, Unsloth**

These are popular open-source tools that facilitate the fine-tuning of LLMs:

*   **Axolotl:**
    *   A powerful and flexible fine-tuning tool built on Hugging Face `transformers`, `peft`, and `accelerate`.
    *   Uses YAML configuration files to define all aspects of the fine-tuning process (base model, dataset, PEFT method, training hyperparameters).
    *   Supports a wide range of models and fine-tuning techniques, including QLoRA, FSDP (Fully Sharded Data Parallel for multi-GPU training).
    *   `[Image: A snippet of an Axolotl YAML configuration file.]`

*   **Apple MLX:**
    *   An array framework for machine learning on Apple silicon (M1, M2, M3 chips), designed by Apple.
    *   It offers a NumPy-like API and an a JAX-like composable function transformations.
    *   The `mlx-examples` repository includes examples for fine-tuning LLMs (like Llama, Mistral) using MLX, optimized for Apple hardware. This allows efficient fine-tuning directly on Macs with Apple Silicon.

*   **Unsloth:**
    *   Specifically optimized for making LoRA/QLoRA fine-tuning significantly faster (often 2x or more) and much more memory-efficient (up to 70% less VRAM) than standard Hugging Face implementations.
    *   Achieves this through custom CUDA kernels and other optimizations.
    *   Makes it possible to fine-tune larger models or use larger batch sizes on consumer GPUs (like RTX 3090/4090).
    *   Very easy to integrate with existing Hugging Face `Trainer` workflows.
    ```python
    from unsloth import FastLlamaModel # or FastMistralModel, etc.
    # ... (load model and tokenizer as usual, but with Unsloth's FastModel)
    model, tokenizer = FastLlamaModel.from_pretrained(
        model_name = "unsloth/llama-2-7b-bnb-4bit", # Example Unsloth pre-quantized model
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    # ... (add LoRA adapters using Unsloth's API)
    model = FastLlamaModel.get_peft_model(
        model,
        r = 16,
        # ... other LoRA config
    )
    # ... (proceed with Hugging Face Trainer)
    ```
    `[Code Block: Simplified Unsloth setup, highlighting its ease of use.]`

**9.9 Post-Finetune Deployment (Ollama Integration)**

Once you've fine-tuned your model, you need to deploy it.
*   **Saving Adapters/Model:**
    *   With PEFT, you often save just the adapter weights (e.g., LoRA layers).
    *   You can also merge these adapters back into the base model to get a full fine-tuned model.
*   **Ollama:**
    *   A popular tool for running LLMs locally on your machine (macOS, Linux, Windows).
    *   It simplifies downloading and running various open-source models, including those in GGUF format.
    *   You can convert your fine-tuned models (especially if merged and quantized to GGUF) to be served via Ollama. This involves creating a "Modelfile" that defines how to run your custom model.
    *   Provides an API endpoint, making it easy to integrate your locally running, fine-tuned LLM into applications.
    `[Image: Ollama logo and a conceptual flow: Fine-Tuned Model -> Convert to GGUF (if needed) -> Ollama Modelfile -> Run with Ollama -> API for Applications.]`

**Chapter Summary:**
Fine-tuning modern LLMs like Llama, Mistral, Gemma, and Phi-3 is now more accessible than ever, thanks to Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA and QLoRA, and user-friendly tools such as Axolotl and Unsloth. From preparing instruction-based datasets to deploying your fine-tuned models with tools like Ollama, this chapter has provided a practical overview of the key steps and considerations involved in customizing these powerful language models.
