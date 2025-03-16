import streamlit as st

def set_page_style():
    """
    Set custom CSS styling for the Streamlit app with light mode
    """
    st.markdown("""
    <style>
        /* Main theme colors - Light Mode */
        :root {
            --primary-color: #3366FF;
            --background-color: #FFFFFF;
            --secondary-background-color: #F5F7F9;
            --text-color: #333333;
            --font: 'Source Sans Pro', sans-serif;
        }
        
        /* Overall page background */
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #111111;
            font-weight: 700;
        }
        
        /* Card-like elements */
        .stCard {
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            background-color: white;
            border: 1px solid #EAEAEA;
            margin-bottom: 15px;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #F5F7F9;
            border-right: 1px solid #EAEAEA;
        }
        
        /* Buttons */
        .stButton button {
            border-radius: 4px;
            font-weight: 600;
            background-color: var(--primary-color);
            color: white;
        }
        
        .stButton button:hover {
            background-color: #2a56d9;
        }
        
        /* Banner image */
        .banner-img {
            border-radius: 8px;
            margin-bottom: 15px;
        }
        
        /* Metric display */
        .metric-container {
            background-color: #FFFFFF;
            border-radius: 6px;
            padding: 12px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            border: 1px solid #EAEAEA;
        }
        
        .metric-value {
            font-size: 20px;
            font-weight: 700;
            color: #3366FF;
        }
        
        .metric-label {
            font-size: 14px;
            color: #555555;
            margin-top: 5px;
        }
        
        /* Make input fields a bit nicer */
        input, textarea, select {
            border-radius: 4px !important;
            border: 1px solid #CCCCCC !important;
        }
        
        /* Improve readability of text elements */
        p, li, label, div {
            color: #333333;
        }
        
        /* Info, success, warning boxes */
        .stAlert {
            border-radius: 4px;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 4px 4px 0 0;
            padding: 8px 16px;
            background-color: #F0F2F6;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: white;
            border-top: 2px solid var(--primary-color);
        }
        
        /* Code blocks */
        code {
            background-color: #F0F2F6;
            color: #333333;
            padding: 2px 4px;
            border-radius: 3px;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background-color: var(--primary-color);
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