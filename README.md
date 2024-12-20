# DocuChat

## Project Overview
This project enables users to interact with PDF documents through natural language questions. Using a RAG (Retrieval-Augmented Generation) pipeline, it provides accurate answers with source attribution, supporting multiple languages.

---

## Installation Steps for Mac and Windows

### Prerequisites
Ensure you have the following installed:
- Python 3.8 or later
- Homebrew (for macOS, optional for managing packages)
- Conda (Download from [Anaconda](https://www.anaconda.com/) for macOS and Windows)

### Steps for macOS
1. Clone the repository:
   ```bash
   git clone https://github.com/FeliGR/DocuChat.git
   cd DocuChat
   ```
2. Create a Conda environment and activate it:
   ```bash
   conda create --name myenv python=3.8
   conda activate myenv
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install Ollama:
   Follow instructions from the [Ollama documentation](https://ollama.ai/install) to install and configure it on your system.

### Steps for Windows
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```
2. Create a Conda environment and activate it:
   ```bash
   conda create --name myenv python=3.8
   conda activate myenv
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install Ollama:
   Follow instructions from the [Ollama documentation](https://ollama.ai/install) to install and configure it on your system.

---

## Architecture Components

### Input
- **PDF Documents**: URLs or files uploaded by the user.

### Processing
- **RAG Pipeline**: Retrieves relevant document sections and generates accurate natural language answers.

### Output
- **Natural Language Answers**: Results include the answer and source attribution.

### Components
- **Data Ingestion**: Handles PDF import and chunking.
- **Search Index**: Optimized for document retrieval.
- **Language Models**: Generates and processes user queries.

---

## Usage Guide

### Access the Web Interface
1. Start the service by running:
   ```bash
   python app.py
   ```
2. Open your browser and go to:
   [http://localhost:7860](http://localhost:7860)

### Perform Actions
- **Enter PDF URLs**: Provide the links to the PDFs you want to analyze.
- **Ask Questions**: Input queries in any language.
- **Get Answers**: Receive responses with detailed source attribution.

---

## Configuration

Update configuration settings in the `config.yaml` file to:
- Adjust chunk sizes.
- Specify language preferences.
- Modify model parameters.

---

## Dependencies

The following dependencies are required:
- `langchain>=0.0.352`
- `gradio==2.9.4`
- `chromadb>=0.4.5`
- `gpt4all>=1.0.8`
- `deep-translator`
- `langdetect`

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## Troubleshooting

### Reset Environment
If you encounter issues, reset the Conda environment:
#### For macOS
```bash
conda remove --name myenv --all
conda create --name myenv python=3.8
conda activate myenv
pip install -r requirements.txt
```

#### For Windows
```bash
conda remove --name myenv --all
conda create --name myenv python=3.8
conda activate myenv
pip install -r requirements.txt
```

### Memory Issues
- Reduce `chunk_size` in `config.yaml`.
- Process fewer documents simultaneously.

---

## License

This project is licensed under the [MIT License](LICENSE).
