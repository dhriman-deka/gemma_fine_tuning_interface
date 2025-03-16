# Gemma Fine-tuning UI

A web-based user interface for fine-tuning Google's Gemma models using Hugging Face infrastructure.

![Gemma Banner](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gemma-banner.png)

## Features

- **Dataset Upload**: Upload and preprocess your custom training data in CSV, JSON, or JSONL format
- **Model Configuration**: Configure Gemma model version and hyperparameters
- **Training Management**: Start, monitor, and manage fine-tuning jobs
- **Evaluation**: Test your fine-tuned model with interactive generation
- **Export Options**: Use your model directly from Hugging Face or export in various formats

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/gemma-finetuning-ui.git
cd gemma-finetuning-ui
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Authentication**: Provide your Hugging Face API token with write permissions
2. **Dataset Preparation**: Upload your dataset and configure column mappings
3. **Model Selection**: Choose between Gemma 2B or 7B and customize training parameters
4. **Training**: Start the fine-tuning process and monitor progress
5. **Evaluation**: Test your fine-tuned model with custom prompts
6. **Deployment**: Export or directly use your model from Hugging Face

## Hugging Face Spaces Deployment

This application is designed to be deployed easily on Hugging Face Spaces:

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces)
2. Select Streamlit as the SDK
3. Connect your GitHub repository or upload the files directly
4. The Space will automatically detect and install the requirements

## Requirements

- Python 3.8+
- Streamlit 1.30.0+
- Hugging Face Account with API token
- For training: GPU access (recommended)

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── pages/              # Multi-page app components
│   ├── 01_Dataset_Upload.py
│   ├── 02_Model_Configuration.py
│   ├── 03_Training_Monitor.py
│   └── 04_Evaluation.py
├── utils/              # Utility functions
│   ├── auth.py         # Authentication utilities
│   ├── huggingface.py  # Hugging Face API integration
│   ├── training.py     # Training utilities
│   └── ui.py           # UI components and styling
├── data/               # Sample data and uploads
└── requirements.txt    # Project dependencies
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Google Gemma Models](https://huggingface.co/google/gemma-7b)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Streamlit](https://streamlit.io/)
- [PEFT - Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)

---

Developed as a simplified interface for fine-tuning Gemma models with Hugging Face. 