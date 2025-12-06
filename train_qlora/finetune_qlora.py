#!/usr/bin/env python3
"""
QLoRA Finetuning Script for PLLuM-12B-chat Model
Task: Text anonymization - replacing sensitive information with placeholders
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import os
from typing import List, Dict

# Configuration
MODEL_NAME = "CYFRAGOVPL/Llama-PLLuM-8B-instruct"
INPUT_FILE = "test.in.txt"
OUTPUT_FILE = "test.out.txt"
OUTPUT_DIR = "./qlora_output"
CHECKPOINT_DIR = "./qlora_checkpoints"

# QLoRA Configuration
LORA_R = 16  # LoRA attention dimension
LORA_ALPHA = 32  # Alpha parameter for LoRA scaling
LORA_DROPOUT = 0.05  # Dropout probability for LoRA layers

# Training Configuration
MAX_LENGTH = 2048  # Maximum sequence length
BATCH_SIZE = 1  # Per device batch size
GRADIENT_ACCUMULATION_STEPS = 16  # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 100
SAVE_STEPS = 50
LOGGING_STEPS = 10


def load_data(input_file: str, output_file: str) -> List[Dict[str, str]]:
    """
    Load input-output pairs from files.
    Each line in input_file corresponds to the same line in output_file.
    """
    print(f"Loading data from {input_file} and {output_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        inputs = f_in.readlines()
    
    with open(output_file, 'r', encoding='utf-8') as f_out:
        outputs = f_out.readlines()
    
    # Ensure both files have the same number of lines
    assert len(inputs) == len(outputs), "Input and output files must have the same number of lines"
    
    data = []
    for inp, out in zip(inputs, outputs):
        inp = inp.strip()
        out = out.strip()
        if inp and out:  # Skip empty lines
            data.append({
                'input': inp,
                'output': out
            })
    
    print(f"Loaded {len(data)} examples")
    return data


def create_prompt(input_text: str) -> str:
    """
    Create a prompt for the model - same format as inference_vllm.py.
    """
    system_message = "Jesteś asystentem specjalizującym się w anonimizacji tekstu. Twoim zadaniem jest zamiana danych osobowych i wrażliwych informacji na odpowiednie znaczniki, takie jak [name], [surname], [phone], [address], [city], [company], [date], [data] itp."
    
    prompt = f"""<|system|>
{system_message}
<|user|>
Zanonimizuj poniższy tekst, zastępując wszystkie dane osobowe i wrażliwe informacje odpowiednimi znacznikami. Przykładowo: "Nazywam się [name] [surname], mój PESEL to [pesel]."

Zwróć wyłącznie zanonimizowany tekst. Żadnych komentarzy ani wyjaśnień.
Wejście do zanonimizowania:
{input_text}
<|assistant|>
Wyjście:
"""
    return prompt


def prepare_dataset(data: List[Dict[str, str]], tokenizer) -> Dataset:
    """
    Prepare dataset for training.
    """
    print("Preparing dataset...")
    
    # Create formatted examples
    formatted_data = []
    for example in data:
        prompt = create_prompt(example['input'])
        full_text = prompt + example['output'] + tokenizer.eos_token
        
        formatted_data.append({
            'text': full_text
        })
    
    # Create dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Tokenize
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        # For causal LM, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    print(f"Dataset prepared with {len(tokenized_dataset)} examples")
    return tokenized_dataset


def setup_model_and_tokenizer():
    """
    Load model with 4-bit quantization and prepare for QLoRA training.
    """
    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model with 4-bit quantization from {MODEL_NAME}...")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    print("Configuring LoRA...")
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer


def train(model, tokenizer, train_dataset):
    """
    Train the model with QLoRA.
    """
    print("Setting up training...")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        fp16=False,
        bf16=True,  # Use bfloat16 for better stability
        optim="paged_adamw_8bit",  # Use 8-bit Adam optimizer
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        report_to="none",  # Change to "tensorboard" or "wandb" if you want logging
        push_to_hub=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train(resume_from_checkpoint=True)
    
    print("Training completed!")
    return trainer


def save_model(model, tokenizer, output_dir):
    """
    Save the fine-tuned model.
    """
    print(f"Saving model to {output_dir}...")
    
    # Save LoRA adapters
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved successfully to {output_dir}")


def main():
    """
    Main training pipeline.
    """
    print("="*50)
    print("QLoRA Finetuning Pipeline")
    print("="*50)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Load data
    data = load_data(INPUT_FILE, OUTPUT_FILE)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Prepare dataset
    train_dataset = prepare_dataset(data, tokenizer)
    
    # Train
    trainer = train(model, tokenizer, train_dataset)
    
    # Save final model
    save_model(model, tokenizer, OUTPUT_DIR)
    
    print("="*50)
    print("Finetuning completed successfully!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
    print("="*50)


if __name__ == "__main__":
    main()

