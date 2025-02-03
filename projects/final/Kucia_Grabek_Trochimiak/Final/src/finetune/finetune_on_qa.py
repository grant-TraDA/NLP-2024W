import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import numpy as np
from typing import Dict
import wandb
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)


def format_instruction(example: Dict) -> str:
    """Format the instruction and response into a single string."""
    instruction = f"### Instruction:\n{example['instruction']}\n\n"
    if example.get('input'):
        instruction += f"### Input:\n{example['input']}\n\n"
    instruction += f"### Response:\n{example['output']}"
    return instruction


def preprocess_function(examples: Dict, tokenizer) -> Dict:
    """Preprocess the examples by tokenizing and formatting."""
    # Format all examples
    formatted_texts = [format_instruction({"instruction": instr, "input": inp, "output": out})
                       for instr, inp, out in zip(examples["instruction"],
                                                  examples["input"],
                                                  examples["output"])]

    # Tokenize
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=2048,
        padding="max_length",
        return_tensors="pt"
    )

    # Create labels (same as input_ids for causal LM)
    tokenized["labels"] = tokenized["input_ids"].clone()

    return tokenized


def create_and_prepare_model(model_name: str):
    """Create and prepare model for 8-bit training with LoRA."""
    # Load model in 8-bit
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Get PEFT model
    model = get_peft_model(model, lora_config)

    return model


def compute_metrics(eval_preds):
    """Compute metrics for evaluation."""
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Calculate accuracy only on non-padded tokens
    mask = labels != -100
    accuracy = (predictions[mask] == labels[mask]).mean()

    return {"accuracy": accuracy}


def train(
        jsonl_path: str,
        model_name: str,
        output_dir: str,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        use_wandb: bool = True
):
    """Main training function."""

    # Initialize wandb if requested
    if use_wandb:
        wandb.init(project="llama-finetune")

    # Load dataset
    dataset = load_dataset("json", data_files=jsonl_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare model
    model = create_and_prepare_model(model_name)

    # Preprocess dataset
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="no", #"steps",
        #eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="wandb" if use_wandb else None,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        #eval_dataset=tokenized_dataset["train"].select(range(min(100, len(tokenized_dataset["train"])))),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Save the final model
    trainer.save_model(os.path.join(output_dir, "final"))

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fine-tune LLaMA model on QA dataset')
    parser.add_argument('jsonl_path', help='Path to the JSONL dataset')
    parser.add_argument('--model_name', default="meta-llama/Llama-3.2-1B-Instruct",
                        help='Name or path of the base model')
    parser.add_argument('--output_dir', default="./llama-finetuned",
                        help='Directory to save the model')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Per device batch size')
    parser.add_argument('--grad_accum', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')

    args = parser.parse_args()

    train(
        jsonl_path=args.jsonl_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        use_wandb=not args.no_wandb
    )
