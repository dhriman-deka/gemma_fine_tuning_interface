import streamlit as st
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils.ui import set_page_style, display_metric
from utils.auth import check_hf_token
from utils.training import simulate_training_progress

# Set page configuration
st.set_page_config(
    page_title="Training Monitor - Gemma Fine-tuning",
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
st.title("📊 Training Monitor")

if not hf_token or not auth_status:
    st.warning("Please authenticate with your Hugging Face token in the sidebar first")
    st.stop()

# Check if training was started
if "model_repo" not in st.session_state:
    st.warning("No active training jobs found")
    st.page_link("pages/02_Model_Configuration.py", label="Go to Model Configuration", icon="⚙️")
    
    # For testing purposes, allow manual entry
    manual_repo = st.text_input("Or enter model repository name manually:")
    if manual_repo:
        st.session_state["model_repo"] = manual_repo
        st.session_state["model_version"] = "google/gemma-2b"  # Default
    else:
        st.stop()

# Training information
st.header("Training Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Repository")
    st.info(st.session_state["model_repo"])
    
    st.subheader("Base Model")
    st.info(st.session_state.get("model_version", "google/gemma-2b"))

with col2:
    # For demo purposes, create a button to start simulated training
    if "training_started" not in st.session_state:
        st.subheader("Start Training")
        if st.button("Launch Training Job", type="primary"):
            st.session_state["training_started"] = True
            st.experimental_rerun()
    else:
        st.subheader("Status")
        st.success("Training in Progress")
        
        # Simulate a cancel button
        if st.button("Cancel Training Job", type="secondary"):
            st.warning("This is a simulation - in a real environment, this would cancel the training job")

# If training has started, show the progress
if st.session_state.get("training_started", False):
    st.header("Training Progress")
    
    # Create a placeholder for the progress bar
    progress_bar = st.progress(0)
    
    # Create placeholder metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Get training progress (simulated for demo)
    progress_data = simulate_training_progress()
    
    # Update progress bar
    progress_bar.progress(progress_data["progress"])
    
    # Update metrics
    with col1:
        display_metric("Epoch", f"{progress_data['current_epoch'] + 1}/{progress_data['total_epochs']}")
    
    with col2:
        display_metric("Loss", f"{progress_data['loss']:.4f}")
    
    with col3:
        display_metric("Learning Rate", f"{progress_data['learning_rate']:.1e}")
    
    with col4:
        status_text = "Complete" if progress_data["status"] == "completed" else "Running"
        display_metric("Status", status_text)
    
    # Create training history visualization
    st.subheader("Training Metrics")
    
    # Simulate training history data
    if "training_history" not in st.session_state:
        st.session_state.training_history = []
    
    # Add current data point to history if not completed
    if progress_data["status"] != "completed" or len(st.session_state.training_history) == 0:
        st.session_state.training_history.append({
            "epoch": progress_data["current_epoch"],
            "loss": progress_data["loss"],
            "learning_rate": progress_data["learning_rate"],
            "timestamp": time.time()
        })
    
    # Convert history to DataFrame
    history_df = pd.DataFrame(st.session_state.training_history)
    
    if not history_df.empty and len(history_df) > 1:
        # Create tabs for different visualizations
        loss_tab, lr_tab = st.tabs(["Loss Curve", "Learning Rate"])
        
        with loss_tab:
            # Create a Plotly figure for the loss curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history_df["epoch"], 
                y=history_df["loss"],
                mode='lines+markers',
                name='Training Loss',
                line=dict(color='#FF4B4B', width=3)
            ))
            
            fig.update_layout(
                title="Training Loss Over Time",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with lr_tab:
            # Create a Plotly figure for the learning rate
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history_df["epoch"], 
                y=history_df["learning_rate"],
                mode='lines+markers',
                name='Learning Rate',
                line=dict(color='#0068C9', width=3)
            ))
            
            fig.update_layout(
                title="Learning Rate Schedule",
                xaxis_title="Epoch",
                yaxis_title="Learning Rate",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Training metrics will appear here once training progresses")
    
    # Training logs
    st.subheader("Training Logs")
    
    # Simulate logs
    log_lines = [
        f"[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}] Initialized training job",
        f"[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading model: {st.session_state.get('model_version', 'google/gemma-2b')}",
        f"[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}] Preparing LoRA configuration"
    ]
    
    # Add epoch logs based on progress
    current_epoch = progress_data["current_epoch"]
    for epoch in range(min(current_epoch + 1, progress_data["total_epochs"])):
        timestamp = pd.Timestamp.now() - pd.Timedelta(seconds=(current_epoch - epoch) * 60)
        log_lines.append(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch+1}/{progress_data['total_epochs']} started")
        
        if epoch < current_epoch:
            # For completed epochs, add completion log
            timestamp = pd.Timestamp.now() - pd.Timedelta(seconds=(current_epoch - epoch - 0.5) * 60)
            sim_loss = max(2.5 - (epoch * 0.5), 0.5)
            log_lines.append(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch+1} completed: loss={sim_loss:.4f}")
    
    # Display logs in a scrollable area
    st.code("\n".join(log_lines))
    
    # Next steps (only show when training is complete)
    if progress_data["status"] == "completed":
        st.success("Training completed successfully!")
        st.page_link("pages/04_Evaluation.py", label="Next: Evaluate Model", icon="🔍")
    else:
        # Auto-refresh for monitoring
        st.empty()
        time.sleep(2)  # Wait for 2 seconds
        st.experimental_rerun() 