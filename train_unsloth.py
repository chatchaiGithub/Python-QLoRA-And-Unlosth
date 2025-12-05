"""
QLoRA Fine-tuning with Unsloth (FIXED)
"""

import torch
import json
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # Disable torch.compile to avoid errors

from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset, DatasetDict

# Configuration
MODEL_NAME = "./Kunoichi-DPO-v2-7B"
OUTPUT_DIR = "./emotional-qlora-output"
DATASET_FILE = "high_quality_training_data_50000.json"

MAX_SEQ_LENGTH = 512
BATCH_SIZE = 2  # Reduced for stability
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
LORA_R = 32
LORA_ALPHA = 16

def format_prompt(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")
    
    return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output_text}"""

def load_dataset_json(file_path, tokenizer, max_length):
    """Load dataset and pre-tokenize to filter out sequences that are too long."""
    print(f"Loading dataset from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Format prompts and filter by token length
    filtered_data = []
    skipped = 0
    
    for item in data:
        text = format_prompt(item)
        # Tokenize to check length
        tokens = tokenizer(text, truncation=False, add_special_tokens=True)
        
        if len(tokens['input_ids']) <= max_length:
            item['text'] = text
            filtered_data.append(item)
        else:
            # Truncate the text to fit
            truncated_tokens = tokenizer(
                text, 
                truncation=True, 
                max_length=max_length,
                add_special_tokens=True
            )
            item['text'] = tokenizer.decode(truncated_tokens['input_ids'], skip_special_tokens=True)
            filtered_data.append(item)
            skipped += 1
    
    print(f"  ✓ Processed {len(filtered_data)} samples ({skipped} truncated)")
    
    split_idx = int(len(filtered_data) * 0.9)
    
    dataset = DatasetDict({
        "train": Dataset.from_list(filtered_data[:split_idx]),
        "validation": Dataset.from_list(filtered_data[split_idx:])
    })
    
    print(f"  ✓ Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")
    return dataset

def main():
    print("="*60)
    print("QLoRA Fine-tuning with Unsloth (FIXED)")
    print("="*60)
    
    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    
    # Ensure tokenizer has proper truncation settings
    tokenizer.truncation_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LENGTH
    
    print("  ✓ Model loaded")
    
    # Load and pre-process dataset with tokenizer
    dataset = load_dataset_json(DATASET_FILE, tokenizer, MAX_SEQ_LENGTH)
    
    print("\nSetting up LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,  # Must be 0 for Unsloth optimization
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=5,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=2,
        warmup_ratio=0.05,
        fp16=True,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        logging_first_step=True,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,  # Disable packing to avoid batch size mismatch
        dataset_num_proc=2,
        truncation=True,  # Enable truncation at the tokenizer level
    )
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    trainer.train()
    
    print("\nSaving model...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    
    print("\n" + "="*60)
    print("✓ Training completed!")
    print(f"  Model saved to: {os.path.join(OUTPUT_DIR, 'final')}")
    print("="*60)

if __name__ == "__main__":
    main()
