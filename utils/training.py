import os
import json
import time
import pandas as pd
import streamlit as st
from huggingface_hub import HfApi, create_repo, upload_file
import tempfile

def create_model_repo(repo_name, private=True):
    """
    Create a new model repository on Hugging Face Hub
    
    Args:
        repo_name (str): Name of the repository
        private (bool): Whether the repository should be private
        
    Returns:
        tuple: (success (bool), message (str))
    """
    try:
        token = st.session_state.get("hf_token")
        if not token:
            return False, "No Hugging Face token found"
        
        username = st.session_state.get("hf_username", "user")
        full_repo_name = f"{username}/{repo_name}"
        
        api = HfApi(token=token)
        repo_url = api.create_repo(
            repo_id=full_repo_name,
            private=private,
            exist_ok=True
        )
        
        return True, repo_url
    except Exception as e:
        return False, str(e)

def upload_training_config(config, repo_name):
    """
    Upload a training configuration file to Hugging Face Hub
    
    Args:
        config (dict): Training configuration
        repo_name (str): Repository to upload to
        
    Returns:
        tuple: (success (bool), message (str))
    """
    try:
        token = st.session_state.get("hf_token")
        if not token:
            return False, "No Hugging Face token found"
            
        username = st.session_state.get("hf_username", "user")
        repo_id = f"{username}/{repo_name}"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
            with open(tmp.name, 'w') as f:
                json.dump(config, f, indent=2)
                
            # Upload file to repository
            upload_file(
                path_or_fileobj=tmp.name,
                path_in_repo="training_config.json",
                repo_id=repo_id,
                token=token
            )
            
            # Clean up temporary file
            tmp_name = tmp.name
        
        os.unlink(tmp_name)
        return True, f"Training config uploaded to {repo_id}"
    except Exception as e:
        return False, str(e)

def setup_training_script(repo_name, config):
    """
    Generate and upload a training script to the repository
    
    Args:
        repo_name (str): Repository name
        config (dict): Training configuration
        
    Returns:
        tuple: (success (bool), message (str))
    """
    # Create a training script using transformers Trainer with 4-bit quantization for CPU
    script = """
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import json
import os
import torch
from huggingface_hub import login
import bitsandbytes as bnb

# Load configuration
with open("training_config.json", "r") as f:
    config = json.load(f)

# Login to Hugging Face
login(token=os.environ.get("HF_TOKEN"))

# Load dataset
print("Loading dataset:", config["dataset_name"])
dataset = load_dataset(config["dataset_name"])

# Prepare train/validation split if not already split
if "train" in dataset and "validation" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.1)
elif "train" not in dataset:
    # If dataset has no train split but has text column, use that
    if "text" in dataset:
        dataset = dataset.train_test_split(test_size=0.1)
    else:
        # Try to find what splits are available
        print("Available splits:", list(dataset.keys()))
        # Default to using the first split and splitting it
        first_split = list(dataset.keys())[0]
        dataset = dataset[first_split].train_test_split(test_size=0.1)

print("Dataset splits:", list(dataset.keys()))

# Print dataset sample
print("Dataset sample:", dataset["train"][0])

# Load tokenizer
print("Loading tokenizer for model:", config["model_name_or_path"])
tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with 4-bit quantization for CPU efficiency
print("Loading model with quantization...")
model = AutoModelForCausalLM.from_pretrained(
    config["model_name_or_path"],
    load_in_4bit=True,  # Enable 4-bit quantization
    device_map="auto",
    quantization_config=bnb.nn.modules.Linear4bit.compute_quant_config(),
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    use_cache=False,  # Required for gradient checkpointing
)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Print memory usage before PEFT
print(f"Model loaded. Memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Prepare model for training with LoRA
print("Setting up LoRA with rank:", config["peft_config"]["r"])
peft_config = LoraConfig(
    r=config["peft_config"]["r"],
    lora_alpha=config["peft_config"]["lora_alpha"],
    lora_dropout=config["peft_config"]["lora_dropout"],
    bias=config["peft_config"]["bias"],
    task_type=config["peft_config"]["task_type"],
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Prepare model - use 8-bit Adam for memory efficiency
print("Preparing model for training...")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Setup training arguments with CPU optimizations
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=config["output_dir"],
    num_train_epochs=config["num_train_epochs"],
    per_device_train_batch_size=config["per_device_train_batch_size"],
    per_device_eval_batch_size=max(1, config["per_device_train_batch_size"] // 2),
    learning_rate=config["learning_rate"],
    weight_decay=config["weight_decay"],
    save_strategy=config["save_strategy"],
    evaluation_strategy=config["evaluation_strategy"],
    fp16=config["fp16"] and torch.cuda.is_available(),
    optim=config["optim"],
    logging_steps=config["logging_steps"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    max_steps=config["max_steps"] if config["max_steps"] > 0 else None,
    warmup_steps=config["warmup_steps"],
    max_grad_norm=config["max_grad_norm"],
    push_to_hub=True,
    hub_token=os.environ.get("HF_TOKEN"),
    dataloader_num_workers=0,  # Lower CPU usage for smaller machines
    use_cpu=not torch.cuda.is_available(),  # Force CPU if no GPU
    lr_scheduler_type="cosine",  # Better LR scheduling for small datasets
    report_to=["tensorboard"],  # Enable tensorboard logging
)

# Define data collator
print("Setting up data collator...")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Tokenize dataset
print("Tokenizing dataset...")
def tokenize_function(examples):
    # Use a smaller max length on CPU to save memory
    max_length = 256 if not torch.cuda.is_available() else 512
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=max_length
    )

# Show progress while tokenizing
print("Mapping tokenization function...")
tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True,
    batch_size=8,  # Smaller batch size for CPU
    remove_columns=dataset["train"].column_names,  # Remove original columns after tokenizing
    desc="Tokenizing dataset",
)

# Initialize trainer
print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
)

# Start training
print("Starting training...")
try:
    trainer.train()
    # Save model and tokenizer
    print("Saving model...")
    trainer.save_model()
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {str(e)}")
    # Save checkpoint even if error occurred
    try:
        trainer.save_model("./checkpoint-error")
        print("Saved checkpoint before error")
    except:
        print("Could not save checkpoint")
"""
    
    try:
        token = st.session_state.get("hf_token")
        if not token:
            return False, "No Hugging Face token found"
            
        username = st.session_state.get("hf_username", "user")
        repo_id = f"{username}/{repo_name}"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as tmp:
            with open(tmp.name, 'w') as f:
                f.write(script)
                
            # Upload file to repository
            upload_file(
                path_or_fileobj=tmp.name,
                path_in_repo="train.py",
                repo_id=repo_id,
                token=token
            )
            
            # Clean up temporary file
            tmp_name = tmp.name
        
        # Also create and upload a CPU-optimized requirements file
        requirements = """
transformers>=4.35.0
peft>=0.7.0
bitsandbytes>=0.40.0
datasets>=2.10.0
torch>=2.0.0
tensorboard>=2.13.0
accelerate>=0.20.0
huggingface_hub>=0.15.0
scipy>=1.10.0
"""
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
            with open(tmp.name, 'w') as f:
                f.write(requirements)
                
            # Upload file to repository
            upload_file(
                path_or_fileobj=tmp.name,
                path_in_repo="requirements.txt",
                repo_id=repo_id,
                token=token
            )
            
            # Clean up temporary file
            tmp_name = tmp.name
        
        os.unlink(tmp_name)
        
        return True, f"Training script and requirements uploaded to {repo_id}"
    except Exception as e:
        return False, str(e)

def simulate_training_progress():
    """
    Simulate training progress for demonstration purposes
    """
    if "training_progress" not in st.session_state:
        st.session_state.training_progress = {
            "started": time.time(),
            "current_epoch": 0,
            "total_epochs": 3,
            "loss": 2.5,
            "learning_rate": 2e-5,
            "progress": 0.0,
            "status": "running"
        }
    
    # Update progress based on elapsed time (simulated)
    elapsed = time.time() - st.session_state.training_progress["started"]
    epoch_duration = 60  # Simulate each epoch taking 60 seconds
    
    # Calculate current progress
    total_duration = epoch_duration * st.session_state.training_progress["total_epochs"]
    progress = min(elapsed / total_duration, 1.0)
    
    # Calculate current epoch
    current_epoch = min(
        int(progress * st.session_state.training_progress["total_epochs"]),
        st.session_state.training_progress["total_epochs"]
    )
    
    # Simulate decreasing loss
    loss = max(2.5 - (progress * 2.0), 0.5)
    
    # Update session state
    st.session_state.training_progress.update({
        "progress": progress,
        "current_epoch": current_epoch,
        "loss": loss,
        "status": "completed" if progress >= 1.0 else "running"
    })
    
    return st.session_state.training_progress 