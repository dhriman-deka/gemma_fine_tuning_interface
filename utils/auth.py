import streamlit as st
import os
from huggingface_hub import HfApi, login

def check_hf_token(token):
    """
    Validate Hugging Face token and login if valid
    
    Args:
        token (str): Hugging Face API token
        
    Returns:
        bool: True if token is valid, False otherwise
    """
    try:
        # Set token in environment and session state
        os.environ["HF_TOKEN"] = token
        st.session_state["hf_token"] = token
        
        # Try to log in
        login(token=token, add_to_git_credential=False)
        
        # Test API access
        api = HfApi(token=token)
        user_info = api.whoami()
        
        # Store username in session state
        st.session_state["hf_username"] = user_info["name"] if "name" in user_info else None
        
        return True
    except Exception as e:
        st.session_state["hf_token"] = None
        st.session_state["hf_username"] = None
        print(f"Authentication error: {str(e)}")
        return False

def get_current_user():
    """
    Get the currently authenticated user's information
    
    Returns:
        dict: User information or None if not authenticated
    """
    if "hf_token" in st.session_state and st.session_state["hf_token"]:
        try:
            api = HfApi(token=st.session_state["hf_token"])
            return api.whoami()
        except:
            return None
    return None 