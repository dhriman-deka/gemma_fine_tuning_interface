import streamlit as st
import pandas as pd
import time
import json
from utils.ui import set_page_style, create_card
from utils.auth import check_hf_token

# Set page configuration
st.set_page_config(
    page_title="Model Evaluation - Gemma Fine-tuning",
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
st.title("🔍 Model Evaluation")

if not hf_token or not auth_status:
    st.warning("Please authenticate with your Hugging Face token in the sidebar first")
    st.stop()

# Check if model is trained
if "model_repo" not in st.session_state:
    st.warning("No trained model found")
    st.page_link("pages/03_Training_Monitor.py", label="Go to Training Monitor", icon="📊")
    
    # For testing purposes, allow manual entry
    manual_repo = st.text_input("Or enter model repository name manually:")
    if manual_repo:
        st.session_state["model_repo"] = manual_repo
        st.session_state["model_version"] = "google/gemma-2b"  # Default
    else:
        st.stop()

# Model information
st.header("Model Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Fine-tuned Model")
    username = st.session_state.get("hf_username", "user")
    model_id = f"{username}/{st.session_state['model_repo']}"
    st.info(model_id)
    
    st.markdown(f"[View on Hugging Face 🔗](https://huggingface.co/{model_id})")

with col2:
    st.subheader("Base Model")
    st.info(st.session_state.get("model_version", "google/gemma-2b"))

# Interactive testing
st.header("Interactive Testing")

def generate_response(prompt, max_length=100, temperature=0.7):
    """Simulate generating a response from the fine-tuned model"""
    # In a real implementation, this would call the model API
    
    # For demo purposes, simulate a response with a delay
    with st.spinner("Generating response..."):
        # Simulate thinking time
        time.sleep(2)
        
        # Generate a simple response based on the prompt for demonstration
        if "hello" in prompt.lower() or "hi" in prompt.lower():
            return "Hello! How can I assist you today?"
        elif "name" in prompt.lower():
            return "I'm a fine-tuned version of Gemma, created to assist you!"
        elif "weather" in prompt.lower():
            return "I don't have real-time data access, but I can tell you about weather patterns and climate information if you'd like."
        elif "help" in prompt.lower():
            return "I'm here to help answer questions, provide information, or assist with tasks. Just let me know what you need!"
        elif len(prompt) < 10:
            return "Could you please provide more details so I can give you a better response?"
        else:
            return f"Based on your request about '{prompt[:20]}...', I've analyzed the content and formulated this response. This is a demonstration of how the fine-tuned model would respond to your input, tailored to the data it was trained on."

# Create the interface
with st.expander("Generation Settings", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        max_length = st.slider("Maximum Length", min_value=10, max_value=500, value=150,
                             help="Maximum number of tokens to generate")
    
    with col2:
        temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1,
                              help="Controls randomness: 1.0 is creative, 0.1 is more deterministic")

prompt = st.text_area("Enter your prompt:", height=150, 
                     help="Enter a prompt or instruction for the model to respond to")

if st.button("Generate Response", type="primary"):
    if prompt:
        response = generate_response(prompt, max_length, temperature)
        
        # Store in history
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        
        st.session_state.conversation_history.append({
            "prompt": prompt,
            "response": response,
            "timestamp": pd.Timestamp.now().isoformat()
        })
        
        # Display response
        st.subheader("Model Response")
        st.markdown(f"<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px;'>{response}</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a prompt first")

# Display conversation history
if "conversation_history" in st.session_state and st.session_state.conversation_history:
    st.header("Conversation History")
    
    for i, item in enumerate(reversed(st.session_state.conversation_history)):
        with st.container():
            st.markdown(f"### Conversation {len(st.session_state.conversation_history) - i}")
            
            st.markdown("**Prompt:**")
            st.markdown(f"<div style='background-color: #e6f7ff; padding: 10px; border-radius: 8px;'>{item['prompt']}</div>", unsafe_allow_html=True)
            
            st.markdown("**Response:**")
            st.markdown(f"<div style='background-color: #f8f9fa; padding: 10px; border-radius: 8px;'>{item['response']}</div>", unsafe_allow_html=True)
            
            st.caption(f"Generated at: {pd.Timestamp(item['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
            st.divider()

# Export section
st.header("Export Model")

export_tab1, export_tab2 = st.tabs(["Export Options", "Usage Guide"])

with export_tab1:
    st.subheader("Export Configuration")
    
    export_format = st.radio("Export Format", options=["Hugging Face Hub", "ONNX", "TensorFlow Lite", "PyTorch"], 
                           horizontal=True)
    
    if export_format == "Hugging Face Hub":
        st.success("Your model is already available on the Hugging Face Hub!")
        username = st.session_state.get("hf_username", "user")
        model_id = f"{username}/{st.session_state['model_repo']}"
        st.code(f"from transformers import AutoModelForCausalLM, AutoTokenizer\n\nmodel = AutoModelForCausalLM.from_pretrained('{model_id}')\ntokenizer = AutoTokenizer.from_pretrained('{model_id}')")
    else:
        st.info("This export format is under development and will be available soon.")
        
        # Show a sample of how to export
        if export_format == "ONNX":
            st.code("""
# Example code for ONNX export (not functional in this demo)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.onnx import export

model_id = "your_username/your_model_repo"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Export to ONNX
export(
    tokenizer=tokenizer,
    model=model,
    output=Path("model.onnx"),
    opset=13
)
            """)
        elif export_format == "PyTorch":
            st.code("""
# Example code for PyTorch export (not functional in this demo)
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "your_username/your_model_repo"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Save model locally
model.save_pretrained("./my_exported_model")
tokenizer.save_pretrained("./my_exported_model")
            """)

with export_tab2:
    st.subheader("How to Use Your Model")
    
    st.markdown("""
    ### Using with Hugging Face Transformers

    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Replace with your model ID
    model_id = "your_username/your_model_repo"
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Generate text
    inputs = tokenizer("What is machine learning?", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
    ```
    
    ### Deployment Options
    
    1. **Hugging Face Inference API** - Easiest option for quick deployment
    2. **Gradio or Streamlit** - For creating interactive demos
    3. **FastAPI or Flask** - For creating backend API services
    4. **Mobile Deployment** - Use TensorFlow Lite or ONNX formats
    """)

# Next steps
st.divider()
st.subheader("Continue Your Journey")

col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/01_Dataset_Upload.py", label="Start New Project", icon="🔄")

with col2:
    st.write("Finished? Export your model or try it out in the playground above.") 