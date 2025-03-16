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
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("✅ **Upload your dataset**")
        st.markdown("✅ **Configure Gemma model parameters**")
        st.markdown("✅ **Start fine-tuning jobs**")
    
    with col2:
        st.markdown("✅ **Monitor training progress**")
        st.markdown("✅ **Evaluate fine-tuned models**") 
        st.markdown("✅ **Export trained models**")

else:
    st.success("You're all set up! Navigate through the pages in the sidebar to start fine-tuning.")
    
    # Quick Start Guide
    st.subheader("Quick Start Guide")
    st.markdown("""
    1. **Upload Dataset** - Prepare and upload your training data
    2. **Configure Model** - Select Gemma version and set hyperparameters
    3. **Train Model** - Start and monitor the training process
    4. **Evaluate & Export** - Test your model and export for deployment
    """)

    # Start buttons
    col1, col2 = st.columns(2)
    with col1:
        st.page_link("pages/01_Dataset_Upload.py", label="Start with Dataset Upload", icon="📤")
    with col2:
        st.page_link("pages/02_Model_Configuration.py", label="Configure Model", icon="⚙️") 