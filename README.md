# 🚀 Fine-Tuning Llama 2 for Q&A with Hugging Face 🤖

This code snippet showcases fine-tuning the Llama 2 large language model (LLM) for a question-and-answer (Q&A) task using the Hugging Face library.

## Key Steps:

### 📦 Installation
- Sets up necessary libraries for working with LLMs, including `transformers` for model handling and `datasets` for managing training data.
- ![1710740975319](https://github.com/CNS-PRADHYUMNA/llama2v2_peft/assets/152390152/1b5b1ba2-1c63-4cf7-b791-0cb1e6d203f0)

### 🧩 Model Loading
- Loads a pre-trained Llama 2 model from the Hugging Face model hub, created by user aboonaaji (`aboonaji/llama2finetune-v2`).
- Parameters: Refers to the number of trainable elements within the LLM that influence its predictions. A higher parameter count often indicates a more complex model capable of learning intricate patterns.
  ![1712936450731](https://github.com/CNS-PRADHYUMNA/llama2v2_peft/assets/152390152/e6515259-e984-41e9-a50a-06bbdef699b7)

### 🔤 Tokenizer Loading
- Retrieves the tokenizer corresponding to the loaded model. This component converts text into numerical sequences the LLM can understand and vice versa.

### ⚙️ Training Arguments
- Defines settings for the training process, such as:
  - `output_dir`: Directory to store training results.
  - `per_device_train_batch_size`: Number of training examples processed simultaneously on a single graphics card (GPU) during each training step.
  - `max_steps`: Maximum number of training steps to perform.

### 🏋️ Trainer Creation
- An `SFTTrainer` object is constructed. This object orchestrates the fine-tuning process. It's provided with:
  - The loaded LLM (model).
  - The training arguments (`args`).
  - The training dataset (`train_dataset`). This dataset should be formatted specifically for the Q&A task, likely containing question-answer pairs.
  - The loaded tokenizer (tokenizer).
  - Peft configuration (`peft_config`): Defines parameters for Low-Rank Adaptation (LORA), a technique for improving efficiency during training.
  - `dataset_text_field`: Specifies the name of the column in the training data that contains the text for training (e.g., "text").

### 📚 Important Terms
- **LORA (Low-Rank Adaptation):** A technique that reduces memory consumption and training time for large models by using a lower-rank representation for certain layers.
- LoRA: Low-Rank Adaptation of Large Language Models
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example -- deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA. We release a package that facilitates the integration of LoRA with PyTorch models and provide our implementations and model checkpoints for RoBERTa, DeBERTa, and GPT-2

  ![1_BX6LodNG9GTfpoGBO2FAuA](https://github.com/CNS-PRADHYUMNA/llama2v2_peft/assets/152390152/2df8b528-3353-4130-b7e0-2d73db75b627)

- **Peft:** A library that implements techniques like LORA to facilitate efficient training of large language models.
- ![1_gUr-3mrzXWeI1UbV4l-kLg](https://github.com/CNS-PRADHYUMNA/llama2v2_peft/assets/152390152/5664598d-e90b-43cf-8fbb-0f0bb57ed58e)


### 🎓 Model Training
- Initiates the training process using the `train()` method on the `SFTTrainer` object. This fine-tunes the Llama 2 model on the provided Q&A dataset.

### 🤝 Interaction with the Model
- Defines a user prompt asking the model's name and about Paracetamol poisoning.
- Creates a text-generation pipeline using the fine-tuned model and tokenizer. This pipeline allows you to generate text from prompts.
- Formats the user prompt with special tokens (`<s>` and `[/INST]`) to instruct the model on how to process the input.
- Feeds the formatted prompt to the text-generation pipeline.
- Prints the generated text response from the fine-tuned model, which should provide information about its name and Paracetamol poisoning.

## 🚦 Techniques for Potentially Faster Fine-Tuning:
- **Pre-trained Llama 2 Model:** Utilizes a pre-trained Llama 2 model, reducing the amount of training needed for the specific medical domain compared to training from scratch.
- **Low-Rank Adaptation (LORA):** Leveraged through the peft library, reduces memory usage and training time by employing a lower-dimensional representation for specific layers within the model.
- **Quantization:** The model might be quantized, reducing the number of bits used to represent data and potentially accelerating computations and lowering memory requirements.
- **Limited Training Steps:** Specifies a maximum number of training steps, controlling the fine-tuning process and saving time.

## 🌟 Considerations
- Actual fine-tuning time depends on various factors including the size and complexity of the medical terms dataset, hardware resources, and specific hyperparameters used.

While the code might not guarantee the absolute fastest fine-tuning time, it incorporates techniques that can contribute to efficient fine-tuning, especially for large models like Llama 2.
