from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig
import torch
import os

# Set Hugging Face API token
os.environ['HUGGINGFACE_API_KEY'] = 'HERE-KEY'

# Configuration parameters
model_name = "bunnycore/LLama-3.1-3b-rp-lora"
dataset_name = "wikitext"
output_dir = "./resultssss"
num_train_epochs = 3
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
optim = "adamw_torch"
save_steps = 1000
logging_steps = 200
learning_rate = 5e-5
weight_decay = 0.01
fp16 = True
bf16 = False
max_grad_norm = 1.0
max_steps = -1
warmup_ratio = 0.1
group_by_length = True
lr_scheduler_type = "linear"
packing = False
max_seq_length = 512
lora_alpha = 16
lora_dropout = 0.1
lora_r = 8
use_4bit = True
bnb_4bit_compute_dtype = "bfloat16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = True
device_map = "auto"


# Load dataset
dataset = load_dataset(dataset_name, 'wikitext-2-raw-v1')
print(dataset)

# Reduce dataset size for training
small_train_dataset = dataset["train"].select(range(100))
small_eval_dataset = dataset["validation"].select(range(10))

print(small_train_dataset)
print(small_eval_dataset)
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ['HUGGINGFACE_API_KEY'])

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_seq_length)

tokenized_train_dataset = small_train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = small_eval_dataset.map(tokenize_function, batched=True)



# Load pre-trained model with quantization configuration for QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_use_double_quant=use_nested_quant,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=os.environ['HUGGINGFACE_API_KEY']
)



# Define the LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["self_attn.k_proj", "self_attn.v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Initialize LoRA with quantized model
model = get_peft_model(model, lora_config)


# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    weight_decay=weight_decay,
    save_steps=save_steps,
    logging_steps=logging_steps,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)


# Custom Data Collator to return loss
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(output_dir)



# Testing the fine-tuned model
test_input = "The history of natural language processing"

# Ensure input tensor is on the correct device
test_input_ids = test_input_ids.to(model.device)

# Create attention mask to avoid warnings during generation
attention_mask = (test_input_ids != tokenizer.pad_token_id).long()

# Generate text with attention mask and device consistency
generated_text = model.generate(
    test_input_ids,
    attention_mask=attention_mask,
    max_length=500
)

# Print the generated output
print(tokenizer.decode(generated_text[0], skip_special_tokens=True))