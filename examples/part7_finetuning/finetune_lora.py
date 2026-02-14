"""
Part 7: LoRA Fine-Tuning with Hugging Face
============================================
Fine-tunes a small pre-trained model (TinyLlama) using LoRA for
parameter-efficient training.

Requires: pip install transformers datasets peft trl accelerate bitsandbytes
"""

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


def main():
    # -------------------------------------------------------------------------
    # Load base model
    # -------------------------------------------------------------------------
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # small enough for a single GPU

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    print(f"Base model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -------------------------------------------------------------------------
    # LoRA configuration — only train a small number of adapter weights
    # -------------------------------------------------------------------------
    lora_config = LoraConfig(
        r=16,                       # rank of the low-rank matrices
        lora_alpha=32,              # scaling factor
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}%)")

    # -------------------------------------------------------------------------
    # Training data (replace with your own dataset)
    # -------------------------------------------------------------------------
    training_data = [
        {"text": "<|user|>\nWhat is gradient descent?\n<|assistant|>\nGradient descent is an optimization algorithm that iteratively adjusts parameters by moving in the direction of steepest decrease of the loss function. The learning rate controls the step size."},
        {"text": "<|user|>\nExplain backpropagation.\n<|assistant|>\nBackpropagation computes gradients of the loss with respect to each parameter by applying the chain rule of calculus backwards through the network, from output layer to input layer."},
        {"text": "<|user|>\nWhat is a transformer?\n<|assistant|>\nA transformer is a neural network architecture that uses self-attention mechanisms to process sequences in parallel. It consists of encoder and/or decoder blocks with multi-head attention and feed-forward layers."},
        # Add hundreds or thousands more examples for real fine-tuning...
    ]

    dataset = Dataset.from_list(training_data)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir="./lora_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,  # set to False if not using NVIDIA GPU; use bf16=True for Ampere+ GPUs
        optim="adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    # Save the LoRA adapter
    model.save_pretrained("./lora_adapter")
    print("LoRA adapter saved to ./lora_adapter")


if __name__ == "__main__":
    main()
