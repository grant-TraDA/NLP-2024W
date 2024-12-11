# mini-rag-bot

------------------------------------------------------------------------------------------------------------------------

### Project structure

```
mini-rag-bot/
│
├── .github/
│   └── workflows/
│       └── code_standard_check.yml     # Workflow for checking code
│
├── data/
│   ├── raw/                            # Raw data (HTML, PDFs, etc.)
│   └── processed/                      # Processed data (extracted text, chunks, etc.)
│
├── modules/
│   ├── data_acquisition/               # For data scraping, file handling, etc.
│   ├── data_processing/                # Text extraction and processing (HTML, PDF, etc.)
│   ├── llm/                            # LLM-related functions (embedding, fine-tuning, etc.)
│   ├── app_logic/                      # Main app logic related to handling queries and managing user interaction
│   └── utils/                          # Utility functions (logging, configuration, etc.)
│
├── deployment/
│   ├── llm/                            # Deployment files for LLM (e.g. Llama3.1 8b)
│   └── embedder/                       # REST API for embedding model
│
├── notebooks/                          # Jupyter notebooks for experiments and prototyping
│
├── tests/                              # Unit tests for various modules
│
├── app.py                              # Main application entry point
├── requirements-dev.txt                # Developer Dependencies
├── requirements.txt                    # Dependencies
└── README.md                           # Project documentation
```

------------------------------------------------------------------------------------------------------------------------

### Environment setup

```bash
conda create -n mini-rag-dev -y python=3.12
conda activate mini-rag-dev
pip install --upgrade pip
pip install -r requirements-dev.txt
pre-commit install
```

### Data scraping

This process take quite long time. We provide processed data as link to data within Google Drive (inside `data/processed` folder).

Run in the project directory:

```bash
python modules/data_acquisition/page_scraping.py --start-url https://ww2.mini.pw.edu.pl/ --data_path ./data
```

### Data preprocessing

Run in the project directory:

```bash
python modules/data_processing/parse_to_text.py
```

### Chunk and embed processed data

Run in the project directory:

```bash
python modules/vector_store/chunk_and_embed.py
```
