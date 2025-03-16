import streamlit as st

def set_page_style():
    """
    Set custom CSS styling for the Streamlit app
    """
    st.markdown("""
    <style>
        /* Main theme colors */
        :root {
            --primary-color: #FF4B4B;
            --background-color: #f5f7f9;
            --secondary-background-color: #FFFFFF;
            --text-color: #262730;
            --font: 'Source Sans Pro', sans-serif;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #1E1E1E;
            font-weight: 700;
        }
        
        /* Card-like elements */
        .stCard {
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            background-color: white;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #FAFAFA;
            border-right: 1px solid #EAEAEA;
        }
        
        /* Buttons */
        .stButton button {
            border-radius: 6px;
            font-weight: 600;
        }
        
        /* Banner image */
        .banner-img {
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        /* Metric display */
        .metric-container {
            background-color: #F9F9F9;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: 700;
            color: #FF4B4B;
        }
        
        .metric-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        
        /* Make input fields a bit nicer */
        input, textarea, select {
            border-radius: 6px !important;
        }
    </style>
    """, unsafe_allow_html=True)

def display_metric(title, value, container=None):
    """
    Display a metric in a nice formatted container
    
    Args:
        title (str): Title of the metric
        value (str/int/float): Value to display
        container (streamlit container, optional): Container to display the metric in
    """
    target = container if container else st
    target.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
    </div>
    """, unsafe_allow_html=True)

def create_card(content, container=None):
    """
    Create a card-like container for content
    
    Args:
        content (callable): Function to call to render content inside the card
        container (streamlit container, optional): Container to place the card in
    """
    target = container if container else st
    
    with target.container():
        target.markdown('<div class="stCard">', unsafe_allow_html=True)
        content()
        target.markdown('</div>', unsafe_allow_html=True) 