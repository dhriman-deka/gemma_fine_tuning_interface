import streamlit as st
import pandas as pd
from utils.ui import set_page_style
from utils.auth import check_hf_token
from utils.huggingface import prepare_training_config
from utils.training import create_model_repo, upload_training_config, setup_training_script

# Set page configuration
st.set_page_config(
    page_title="Model Configuration - Gemma Fine-tuning",
    page_icon="🤖",
    layout="wide"
)

# Apply custom styling
set_page_style()

# Sidebar for authentication
with st.sidebar:
    st.title("🤖 Gemma Fine-tuning")
    st.image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gemma-banner.png", 
             use_column_width=True)
    
    # Authentication section
    st.subheader("🔑 Authentication")
    hf_token = st.text_input("Hugging Face API Token", type="password", 
                             help="Enter your Hugging Face write token to enable model fine-tuning")
    auth_status = check_hf_token(hf_token) if hf_token else False
    
    if auth_status:
        st.success("Authenticated successfully!")
    elif hf_token:
        st.error("Invalid token. Please check and try again.")
    
    st.divider()
    st.caption("A simple UI for fine-tuning Gemma models")

# Main content
st.title("⚙️ Model Configuration")

if not hf_token or not auth_status:
    st.warning("Please authenticate with your Hugging Face token in the sidebar first")
    st.stop()

# Check if dataset repository is set
if "dataset_repo" not in st.session_state:
    st.warning("Please upload a dataset first")
    st.page_link("pages/01_Dataset_Upload.py", label="Go to Dataset Upload", icon="📤")
    
    # For testing purposes, allow manual entry
    manual_repo = st.text_input("Or enter dataset repository name manually:")
    if manual_repo:
        st.session_state["dataset_repo"] = manual_repo
    else:
        st.stop()

# Model configuration
st.header("1. Select Gemma Model")

model_version = st.radio(
    "Select Gemma model version",
    options=["google/gemma-2b", "google/gemma-7b"],
    horizontal=True,
    help="Choose the Gemma model size. 2B is faster, 7B is more capable."
)

st.header("2. Training Configuration")

# Output repository settings
with st.expander("Output Repository Settings", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        output_repo_name = st.text_input(
            "Repository Name",
            value=f"gemma-finetuned-{pd.Timestamp.now().strftime('%Y%m%d')}",
            help="Name of the Hugging Face repository to store your fine-tuned model"
        )
    
    with col2:
        is_private = st.toggle("Private Repository", value=True, 
                               help="Make your model repository private (recommended)")

# Tabs for Basic and Advanced configuration
basic_tab, advanced_tab = st.tabs(["Basic Configuration", "Advanced Configuration"])

with basic_tab:
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Number of Epochs", min_value=1, max_value=10, value=3, 
                          help="Number of complete passes through the dataset")
        
        batch_size = st.slider("Batch Size", min_value=1, max_value=32, value=8,
                             help="Number of examples processed together")
    
    with col2:
        learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-3, 
                                       value=2e-5, format="%.1e",
                                       help="Step size for gradient updates")
        
        fp16_training = st.toggle("Use FP16 Training", value=True,
                                help="Use 16-bit precision for faster training")

with advanced_tab:
    col1, col2 = st.columns(2)
    
    with col1:
        lora_rank = st.slider("LoRA Rank", min_value=1, max_value=64, value=8, 
                            help="Rank of the LoRA matrices")
        
        lora_alpha = st.slider("LoRA Alpha", min_value=1, max_value=128, value=32,
                             help="Scaling factor for LoRA")
        
        lora_dropout = st.slider("LoRA Dropout", min_value=0.0, max_value=0.5, value=0.05, step=0.01,
                               help="Dropout probability for LoRA layers")
    
    with col2:
        weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=0.1, 
                                      value=0.01, step=0.001,
                                      help="L2 regularization strength")
        
        gradient_accumulation = st.slider("Gradient Accumulation Steps", min_value=1, max_value=16, value=1,
                                        help="Number of steps to accumulate gradients")
        
        warmup_steps = st.slider("Warmup Steps", min_value=0, max_value=500, value=0,
                               help="Steps of linear learning rate warmup")

st.header("3. Training Summary")

# Create hyperparameters dictionary
hyperparams = {
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "fp16": fp16_training,
    "lora_rank": lora_rank,
    "lora_alpha": lora_alpha,
    "lora_dropout": lora_dropout,
    "weight_decay": weight_decay,
    "gradient_accumulation": gradient_accumulation,
    "warmup_steps": warmup_steps,
    "max_steps": -1,  # -1 means train for full epochs
    "max_grad_norm": 1.0
}

# Display summary
col1, col2 = st.columns(2)

with col1:
    st.subheader("Selected Model")
    st.info(model_version)
    
    st.subheader("Dataset Repository")
    st.info(st.session_state["dataset_repo"])

with col2:
    st.subheader("Key Hyperparameters")
    st.write(f"Epochs: {epochs}")
    st.write(f"Batch Size: {batch_size}")
    st.write(f"Learning Rate: {learning_rate}")
    st.write(f"LoRA Rank: {lora_rank}")

# Start training button
st.header("4. Start Training")

if st.button("Prepare and Launch Training", type="primary"):
    with st.spinner("Setting up training job..."):
        # First create the model repository
        success, repo_url = create_model_repo(output_repo_name, private=is_private)
        
        if not success:
            st.error(f"Failed to create model repository: {repo_url}")
        else:
            st.success(f"Created model repository: {output_repo_name}")
            
            # Prepare training configuration
            config = prepare_training_config(
                model_name=model_version,
                hyperparams=hyperparams,
                dataset_repo=st.session_state["dataset_repo"],
                output_repo=output_repo_name
            )
            
            # Upload training configuration
            success, message = upload_training_config(config, output_repo_name)
            
            if not success:
                st.error(f"Failed to upload training configuration: {message}")
            else:
                st.success("Uploaded training configuration")
                
                # Setup training script
                success, message = setup_training_script(output_repo_name, config)
                
                if not success:
                    st.error(f"Failed to setup training script: {message}")
                else:
                    st.success("Uploaded training script")
                    
                    # Store in session state
                    st.session_state["model_repo"] = output_repo_name
                    st.session_state["model_version"] = model_version
                    st.session_state["hyperparams"] = hyperparams
                    
                    # Success message and next step
                    st.success("Training job prepared successfully! You can now monitor the training progress.")
                    st.page_link("pages/03_Training_Monitor.py", label="Next: Monitor Training", icon="📊") 