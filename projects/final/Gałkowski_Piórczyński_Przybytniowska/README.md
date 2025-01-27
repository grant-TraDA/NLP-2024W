# MiNI RAG Bot

Authors:
- [Mikołaj Gałkowski](https://github.com/galkowskim)
- [Mikołaj Piórczyński](https://github.com/mpiorczynski)
- [Julia Przybytniowska](https://github.com/przybytniowskaj)

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
│   └── evaluation/                     # Evaluation-related files
│
├── deployment/
│   ├── llm/                            # Deployment files for LLM (e.g. Llama3.1 8b)
│   └── embedder/                       # REST API for embedding model
│
├── notebooks/                          # Jupyter notebooks for experiments and prototyping and exploratory data analysis
│
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


## Project reproduction

0. Set up environment (see above).
1. Data scraping + preprocessing (you can download preprocessed data from [Google Drive](https://drive.google.com/drive/u/3/folders/1NubcW8_F46ftfeUlZxNYvVPI2A7L8r2y)).
2. Chunk and embed processed data.
   - in order to do that please deploy or run locally the embedding model (see `deployment/embedder` folder)
   - set up constants in `modules/vector_store/chunk_and_embed.py` script
3. Deploy or run locally the LLM model (see `deployment/llm` folder).
4. Run Streamlit app using `streamlit run app.py`.

> Below we present more detailed steps of different steps
----------------------------------------------------------------------------------------



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

Please firstly look into the script and correctly set-up constants. If you intend to use embedding model, which can't be run locally, you can use REST API script provided in `deployment/embedder` folder, use it to deploy the model on a machine with GPU and set the `EMBEDDER_URL` constant in the script to the correct URL.

Run in the project directory:

```bash
python modules/vector_store/chunk_and_embed.py
```

### Run Streamlit app

Run in the project directory:

```bash
streamlit run app.py
```

### Evaluate models performance using RAGAS framework

Please set up constants in the script to match your environment. (`LLM_API_URL`, `LLM_MODEL_NAME`, `EMBEDDER_MODEL_NAME`) Also ensure that you have

```bash
python -m modules.evaluation.ragas_evaluation
```
