# RAG ChatBot

This project is a Retrieval-Augmented Generation (RAG) Chatbot for Scientific Papers designed to assist users in navigating, summarizing, and understanding scientific literature. Built as a web app, this tool allows users to ask complex questions about scientific content, receive precise answers, and engage with large volumes of research material more interactively and efficiently.

## Setup

1. Install dependencies: ```pip install -r ChatBot/requirements.txt```
2. Install Ollama - https://ollama.com/download
3. Download LLM - ```ollama pull qwen2.5:7b-instruct-q4_0```
4. Run app: ```python -m ChatBot/streamlit run app.py```

## Technical Stack
 
 - **Frontend:** Streamlit (Python-based web app framework for simplicity and speed).
 - **Backend:** LangChain for modular integration with LLMs and retrieval components.
 - **LLM Serving:**: Ollama, enabling efficient model deployment and performance.

## Customization

We’ve included a Python script, `pdf2faiss.py`, which allows users to create their own vector stores and customize the chatbot to suit their specific needs. Once the vector store is created, simply update its path in `app.py` on line 17.
