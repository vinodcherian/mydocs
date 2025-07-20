#Part 3: Fine-Tuning Language Models in Practice

#Chapter 6: Fine-Tuning Classical Language Models (BERT, T5)

Before the era of massive generative LLMs like Llama or GPT-4, "classical" Transformer-based models like BERT and T5 were state-of-the-art for many NLP tasks. Fine-tuning these models is still relevant for specific applications and provides a great learning ground for understanding the core fine-tuning process.

**6.1 BERT Fine-Tuning (Text Classification / QA)**

*   **BERT (Bidirectional Encoder Representations from Transformers):**
    *   An encoder-only Transformer model.
    *   Pre-trained on two tasks: Masked Language Modeling (predicting masked words) and Next Sentence Prediction.
    *   Excellent for tasks requiring a deep understanding of context, like text classification, question answering, and named entity recognition.

*   **Fine-Tuning BERT for Text Classification (e.g., Sentiment Analysis):**
    1.  **Load Pre-trained BERT Model and Tokenizer:**
        ```python
        from transformers import BertTokenizer, BertForSequenceClassification
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2) # e.g., for positive/negative sentiment
        ```
    2.  **Prepare Your Dataset:**
        *   Sentences (or documents) and their corresponding labels (e.g., "positive", "negative").
        *   Tokenize the text using the BERT tokenizer.
            `[Diagram: Text -> Tokenizer -> Input IDs, Attention Mask, Token Type IDs]`
    3.  **Training Loop (Simplified with Hugging Face Trainer):**
        *   The `Trainer` class handles most of the training details (batching, optimizer, learning rate scheduler, evaluation).
        ```python
        from transformers import TrainingArguments, Trainer

        training_args = TrainingArguments(
            output_dir='./results_bert_sentiment',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            # ... other arguments
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset, # Your tokenized training data
            eval_dataset=tokenized_eval_dataset,   # Your tokenized evaluation data
        )
        trainer.train()
        ```
        `[Code Block: Simplified Hugging Face Trainer setup for BERT classification.]`
    4.  **How it works:** The pre-trained BERT model's general language understanding is adapted. The added classification head (a simple neural network layer on top of BERT's output) is trained to map BERT's contextual representations to your specific labels.

*   **Fine-Tuning BERT for Question Answering (e.g., SQuAD-style):**
    1.  **Load `BertForQuestionAnswering`:** This model has a head designed to predict the start and end tokens of the answer span within a given context.
    2.  **Dataset Format:** Context paragraph, question, and the start/end character positions of the answer in the context.
    3.  The fine-tuning process adapts BERT to identify relevant answer spans based on the question and context.

**6.2 T5 Fine-Tuning (Text2Text Tasks)**

*   **T5 (Text-to-Text Transfer Transformer):**
    *   An encoder-decoder Transformer model.
    *   Unique Approach: T5 frames *every* NLP task as a "text-to-text" problem. For example:
        *   Translation: "translate English to German: That is good." -> "Das ist gut."
        *   Summarization: "summarize: [long article text]" -> "[short summary]"
        *   Classification: "cola sentence: The movie was great." -> "positive"
    *   This unified framework makes it very versatile.

*   **Fine-Tuning T5:**
    1.  **Load Pre-trained T5 Model and Tokenizer:**
        ```python
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        ```
    2.  **Prepare Your Dataset:**
        *   Input text (often prefixed with a task description like "translate English to French: ") and target output text.
        *   Tokenize both inputs and outputs.
    3.  **Training Loop (using Hugging Face Trainer):** Similar to BERT, but the model generates text as output. The loss is calculated based on how well the generated text matches the target text.
        `[Code Block: Simplified Hugging Face Trainer setup for T5 (mentioning DataCollatorForSeq2Seq).]`
    4.  **Example Tasks for T5 Fine-Tuning:**
        *   Summarization: Fine-tune on article-summary pairs.
        *   Translation: Fine-tune on parallel sentences in two languages.
        *   Paraphrasing: Fine-tune on sentence-paraphrase pairs.
        *   Instruction Following: Fine-tune on instruction-response pairs.

**Key Differences in Fine-Tuning BERT vs. T5:**

*   **Architecture:** BERT is encoder-only (good for understanding, outputs fixed-size representations or classifications). T5 is encoder-decoder (good for generation, outputs variable-length text).
*   **Task Formulation:** BERT often requires task-specific heads. T5 treats all tasks as text-to-text.
*   **Output:** BERT's fine-tuning for classification gives class probabilities. T5 generates output text.

**Chapter Summary:**
Fine-tuning classical models like BERT and T5 is a valuable exercise. BERT excels at understanding-based tasks like classification and QA by adapting its powerful encoder. T5's text-to-text framework makes it highly versatile for a range of generative and understanding tasks, fine-tuning its encoder-decoder architecture to map input text prompts to desired output text. The Hugging Face `Trainer` API greatly simplifies the fine-tuning process for both.
