# Artistic Chatbot

## Installation
As a prerequisite, you need to have installed Hugging Face's CLI and you need to have access to `meta-llama` repository.

You also need to have OpenAI API key saved to environmental variable `OPENAI_API_KEY`.

Finally, you need to install the requirements:
```bash
pip install -r requirements.txt
```

## Usage
You need to first create Chroma DB from the existing text sources. Check `src/rag/add_to_db.py` for more information.
Set the name of collection to be `rag-collection`.

To start the chatbot, run:
```bash
python src/app.py
```
