import streamlit as st
from utils.auth import check_hf_token
from utils.ui import set_page_style

# Set page configuration
st.set_page_config(
    page_title="Gemma Fine-tuning UI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
set_page_style()

# Sidebar for navigation and authentication
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
    st.markdown("**Navigation:**")
    if auth_status:
        st.page_link("pages/01_Dataset_Upload.py", label="1️⃣ Dataset Upload", icon="📤")
        st.page_link("pages/02_Model_Configuration.py", label="2️⃣ Model Configuration", icon="⚙️")
        st.page_link("pages/03_Training_Monitor.py", label="3️⃣ Training Monitor", icon="📊")
        st.page_link("pages/04_Evaluation.py", label="4️⃣ Model Evaluation", icon="🔍")
    
    st.caption("A simple UI for fine-tuning Gemma models")

# Main content
st.title("Welcome to Gemma Fine-tuning UI")

if not hf_token:
    st.info("👈 Please enter your Hugging Face token in the sidebar to get started")
    
    with st.expander("ℹ️ How to get a Hugging Face token", expanded=True):
        st.markdown("""
        1. Go to [Hugging Face](https://huggingface.co/settings/tokens)
        2. Sign in or create an account
        3. Create a new token with write access
        4. Copy and paste the token in the sidebar
        """)
    
    st.divider()
    
    st.subheader("What you can do with this app:")
    
    st.markdown("""
    ### 📝 Simple Fine-tuning Process
    
    This app provides a straightforward interface for fine-tuning Gemma models with your own data:
    """)
    
    cols = st.columns(2)
    
    with cols[0]:
        st.markdown("""
        ✅ **Upload your dataset**
        - Support for CSV and JSON/JSONL formats
        - Manual input option for small datasets
        - Automatic preprocessing for Gemma format
        
        ✅ **Configure Gemma model parameters**
        - Choose between Gemma 2B and 7B models
        - Adjust learning rate, batch size, and epochs
        - LoRA parameter configuration
        """)
    
    with cols[1]:
        st.markdown("""
        ✅ **Monitor training progress**
        - Visual training progress tracking
        - Loss curve visualization
        - Training logs and status updates
        
        ✅ **Evaluate and use your model**
        - Interactive testing interface
        - Export options for deployment
        - Usage examples with code snippets
        """)

else:
    st.success("You're all set up! Follow these steps to fine-tune your Gemma model:")
    
    # Simple step-by-step guide
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Step 1: Prepare Your Dataset")
        st.markdown("""
        - Upload your dataset in CSV or JSONL format
        - Ensure your data has prompt/instruction and response columns
        - The app will preprocess the data into the right format for Gemma
        """)
        st.page_link("pages/01_Dataset_Upload.py", label="Go to Dataset Upload", icon="📤")
    
    with col2:
        st.markdown("### Step 2: Configure Your Model")
        st.markdown("""
        - Select either Gemma 2B (faster) or 7B (more powerful)
        - Adjust hyperparameters based on your needs
        - Basic configurations work well for most use cases
        """)
        st.page_link("pages/02_Model_Configuration.py", label="Go to Model Configuration", icon="⚙️")
    
    st.divider()
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### Step 3: Train Your Model")
        st.markdown("""
        - Start the training process
        - Monitor progress with real-time updates
        - Training a model may take time depending on your dataset size
        """)
        st.page_link("pages/03_Training_Monitor.py", label="Go to Training Monitor", icon="📊")
    
    with col4:
        st.markdown("### Step 4: Evaluate & Use Your Model")
        st.markdown("""
        - Test your model with custom prompts
        - Compare results with the base model
        - Export your model for use in applications
        """)
        st.page_link("pages/04_Evaluation.py", label="Go to Model Evaluation", icon="🔍")
    
    # Notes about CPU limitations
    st.info("""
    **Note on Training Limitations**: This app is running on CPU resources (2vCPU, 16GB RAM), which means:
    
    - For actual training, we use Parameter-Efficient Fine-Tuning (PEFT) with LoRA to reduce memory requirements
    - Training will be slower than on GPU hardware
    - For very large datasets, consider using this interface to prepare your data and configuration, 
      then download the config to run training on more powerful hardware
    """) 