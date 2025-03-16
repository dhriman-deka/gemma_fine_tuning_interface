import os
import json
import tempfile
import pandas as pd
import streamlit as st
from huggingface_hub import HfApi, upload_file, create_repo
from transformers import AutoTokenizer

def create_dataset_repo(repo_name, private=True):
    """
    Create a new dataset repository on Hugging Face Hub
    
    Args:
        repo_name (str): Name of the repository
        private (bool): Whether the repository should be private
        
    Returns:
        str: URL of the created repository
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
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
        
        return True, repo_url
    except Exception as e:
        return False, str(e)

def upload_dataset_to_hub(file_data, file_name, repo_name):
    """
    Upload a dataset file to Hugging Face Hub
    
    Args:
        file_data (bytes/DataFrame): File content as bytes or a DataFrame
        file_name (str): Name to save the file as
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
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp:
            # If it's a DataFrame, save as JSONL
            if isinstance(file_data, pd.DataFrame):
                file_data.to_json(tmp.name, orient="records", lines=True)
            else:
                # Otherwise, assume it's bytes
                tmp.write(file_data)
                
            # Upload file to repository
            upload_file(
                path_or_fileobj=tmp.name,
                path_in_repo=file_name,
                repo_id=repo_id,
                token=token,
                repo_type="dataset"
            )
            
            # Clean up temporary file
            tmp_name = tmp.name
        
        os.unlink(tmp_name)
        return True, f"File uploaded to {repo_id}"
    except Exception as e:
        return False, str(e)

def prepare_training_config(model_name, hyperparams, dataset_repo, output_repo):
    """
    Prepare a training configuration for Gemma fine-tuning
    
    Args:
        model_name (str): Model identifier
        hyperparams (dict): Training hyperparameters
        dataset_repo (str): Dataset repository name
        output_repo (str): Output repository name
        
    Returns:
        dict: Training configuration
    """
    username = st.session_state.get("hf_username", "user")
    
    config = {
        "model_name_or_path": model_name,
        "dataset_name": f"{username}/{dataset_repo}",
        "output_dir": f"{username}/{output_repo}",
        "num_train_epochs": hyperparams.get("epochs", 3),
        "per_device_train_batch_size": hyperparams.get("batch_size", 8),
        "learning_rate": hyperparams.get("learning_rate", 2e-5),
        "weight_decay": hyperparams.get("weight_decay", 0.01),
        "save_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "fp16": hyperparams.get("fp16", False),
        "peft_config": {
            "r": hyperparams.get("lora_rank", 8),
            "lora_alpha": hyperparams.get("lora_alpha", 32),
            "lora_dropout": hyperparams.get("lora_dropout", 0.05),
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        "optim": "adamw_torch",
        "logging_steps": 50,
        "gradient_accumulation_steps": hyperparams.get("gradient_accumulation", 1),
        "max_steps": hyperparams.get("max_steps", -1),
        "warmup_steps": hyperparams.get("warmup_steps", 0),
        "max_grad_norm": hyperparams.get("max_grad_norm", 1.0),
    }
    
    return config

def preprocess_dataset(df, prompt_column, response_column, model_name="google/gemma-2b"):
    """
    Preprocess a dataset for Gemma fine-tuning
    
    Args:
        df (DataFrame): Dataset
        prompt_column (str): Column containing prompts/instructions
        response_column (str): Column containing responses
        model_name (str): Model identifier for tokenizer
        
    Returns:
        DataFrame: Processed dataset
    """
    # Check if columns exist
    if prompt_column not in df.columns or response_column not in df.columns:
        raise ValueError(f"Columns {prompt_column} and/or {response_column} not found in dataset")
    
    # Simple format for instruction tuning
    df["text"] = df.apply(
        lambda row: f"<start_of_turn>user\n{row[prompt_column]}<end_of_turn>\n<start_of_turn>model\n{row[response_column]}<end_of_turn>", 
        axis=1
    )
    
    # Return the processed dataset
    return df[["text"]] 