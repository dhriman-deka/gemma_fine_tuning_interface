import streamlit as st
import pandas as pd
import io
import json
import os
from utils.ui import set_page_style
from utils.auth import check_hf_token
from utils.huggingface import create_dataset_repo, upload_dataset_to_hub, preprocess_dataset

# Set page configuration
st.set_page_config(
    page_title="Dataset Upload - Gemma Fine-tuning",
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
st.title("📤 Dataset Upload")

if not hf_token or not auth_status:
    st.warning("Please authenticate with your Hugging Face token in the sidebar first")
    st.stop()

# Dataset repository settings
with st.expander("Dataset Repository Settings", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        dataset_repo_name = st.text_input(
            "Repository Name",
            value=f"gemma-finetune-dataset-{pd.Timestamp.now().strftime('%Y%m%d')}",
            help="Name of the Hugging Face repository to store your dataset"
        )
    
    with col2:
        is_private = st.toggle("Private Repository", value=True, 
                               help="Make your dataset repository private (recommended)")

# Dataset upload section
st.header("Upload Your Dataset")

dataset_format = st.radio(
    "Select dataset format",
    options=["CSV", "JSON/JSONL", "Manual Input"],
    horizontal=True
)

# Session state to store dataset
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "dataset_preview" not in st.session_state:
    st.session_state.dataset_preview = None

# Upload handlers
if dataset_format == "CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file:
        try:
            # Read the file
            df = pd.read_csv(uploaded_file)
            st.session_state.dataset = df
            st.session_state.dataset_preview = df.head(5)
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

elif dataset_format == "JSON/JSONL":
    uploaded_file = st.file_uploader("Upload a JSON/JSONL file", type=["json", "jsonl"])
    
    if uploaded_file:
        try:
            # Read file contents
            content = uploaded_file.read()
            
            # Try to parse as JSON array first
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame([data])
            except:
                # Try to parse as JSONL
                try:
                    content = io.StringIO(content.decode("utf-8"))
                    df = pd.read_json(content, lines=True)
                except:
                    raise ValueError("Invalid JSON/JSONL format")
            
            st.session_state.dataset = df
            st.session_state.dataset_preview = df.head(5)
        except Exception as e:
            st.error(f"Error reading JSON/JSONL file: {str(e)}")

elif dataset_format == "Manual Input":
    st.info("Enter your training examples manually:")
    
    num_examples = st.number_input("Number of examples", min_value=1, max_value=20, value=3)
    examples = []
    
    for i in range(num_examples):
        st.subheader(f"Example {i+1}")
        col1, col2 = st.columns(2)
        
        with col1:
            prompt = st.text_area(f"Prompt/Instruction {i+1}", height=100)
        
        with col2:
            response = st.text_area(f"Response {i+1}", height=100)
        
        if prompt and response:
            examples.append({"prompt": prompt, "response": response})
    
    if examples:
        df = pd.DataFrame(examples)
        st.session_state.dataset = df
        st.session_state.dataset_preview = df

# Display dataset preview if available
if st.session_state.dataset_preview is not None:
    st.subheader("Dataset Preview")
    st.dataframe(st.session_state.dataset_preview, use_container_width=True)
    
    # Column mapping
    st.subheader("Column Mapping")
    st.info("Select which columns contain prompts/instructions and responses")
    
    columns = st.session_state.dataset.columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        prompt_column = st.selectbox("Prompt/Instruction Column", options=columns, 
                                    index=columns.index("prompt") if "prompt" in columns else 0)
    
    with col2:
        response_column = st.selectbox("Response Column", options=columns,
                                      index=columns.index("response") if "response" in columns else min(1, len(columns)-1))
    
    # Dataset statistics
    st.subheader("Dataset Statistics")
    
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    with stats_col1:
        st.metric("Total Examples", len(st.session_state.dataset))
    
    with stats_col2:
        avg_prompt_len = st.session_state.dataset[prompt_column].str.len().mean().round(1)
        st.metric("Avg. Prompt Length", f"{avg_prompt_len} chars")
    
    with stats_col3:
        avg_response_len = st.session_state.dataset[response_column].str.len().mean().round(1)
        st.metric("Avg. Response Length", f"{avg_response_len} chars")
    
    # Upload button
    st.subheader("Upload to Hugging Face")
    
    if st.button("Process and Upload Dataset", type="primary"):
        with st.spinner("Processing dataset..."):
            # Preprocess the dataset into the right format for Gemma
            try:
                processed_df = preprocess_dataset(
                    st.session_state.dataset, 
                    prompt_column=prompt_column,
                    response_column=response_column
                )
                
                # Create repository
                success, result = create_dataset_repo(dataset_repo_name, private=is_private)
                
                if not success:
                    st.error(f"Failed to create repository: {result}")
                else:
                    st.success(f"Created repository: {dataset_repo_name}")
                    
                    # Upload dataset
                    success, result = upload_dataset_to_hub(
                        processed_df, 
                        "train.jsonl", 
                        dataset_repo_name
                    )
                    
                    if success:
                        st.session_state["dataset_repo"] = dataset_repo_name
                        st.success(f"Dataset uploaded successfully! You can now proceed to model configuration.")
                        
                        # Next steps button
                        st.page_link("pages/02_Model_Configuration.py", label="Next: Configure Model", icon="⚙️")
                    else:
                        st.error(f"Failed to upload dataset: {result}")
            except Exception as e:
                st.error(f"Error processing dataset: {str(e)}")
else:
    st.info("Please upload or input your dataset above") 