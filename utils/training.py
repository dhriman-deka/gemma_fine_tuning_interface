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
    # Create a training script using transformers Trainer
    script = """
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import os
import torch
from huggingface_hub import login

# Load configuration
with open("training_config.json", "r") as f:
    config = json.load(f)

# Login to Hugging Face
login(token=os.environ.get("HF_TOKEN"))

# Load dataset
dataset = load_dataset(config["dataset_name"])

# Prepare train/validation split if not already split
if "train" in dataset and "validation" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.1)
elif "train" not in dataset:
    dataset = dataset["text"].train_test_split(test_size=0.1)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
model = AutoModelForCausalLM.from_pretrained(
    config["model_name_or_path"], 
    torch_dtype=torch.float16 if config.get("fp16", False) else torch.float32,
    device_map="auto"
)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Prepare model for training with LoRA
peft_config = LoraConfig(
    r=config["peft_config"]["r"],
    lora_alpha=config["peft_config"]["lora_alpha"],
    lora_dropout=config["peft_config"]["lora_dropout"],
    bias=config["peft_config"]["bias"],
    task_type=config["peft_config"]["task_type"],
    target_modules=["q_proj", "v_proj"]  # Adjust target modules based on your model
)

# Prepare model
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# Setup training arguments
training_args = TrainingArguments(
    output_dir=config["output_dir"],
    num_train_epochs=config["num_train_epochs"],
    per_device_train_batch_size=config["per_device_train_batch_size"],
    learning_rate=config["learning_rate"],
    weight_decay=config["weight_decay"],
    save_strategy=config["save_strategy"],
    evaluation_strategy=config["evaluation_strategy"],
    fp16=config["fp16"],
    optim=config["optim"],
    logging_steps=config["logging_steps"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    max_steps=config["max_steps"] if config["max_steps"] > 0 else None,
    warmup_steps=config["warmup_steps"],
    max_grad_norm=config["max_grad_norm"],
    push_to_hub=True,
    hub_token=os.environ.get("HF_TOKEN")
)

# Define data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save model and tokenizer
trainer.save_model()
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
        
        os.unlink(tmp_name)
        return True, f"Training script uploaded to {repo_id}"
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