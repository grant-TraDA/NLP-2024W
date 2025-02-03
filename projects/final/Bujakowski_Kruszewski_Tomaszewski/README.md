This project provides an **interactive research assistant** designed to enhance your document comprehension and navigation. The assistant allows users to query multiple PDFs for specific answers, retrieving relevant document sections. It includes an intuitive interface for viewing and navigating to specific pages of PDFs or PowerPoint files, offering targeted previews for improved understanding.

## Getting Started

### Prerequisites

Ensure you have all required dependencies installed. You can do this by running:
```bash
pip install -r requirements.txt
```

### Setup

1. Place the `vector_store` directory (for storing vectorized document representations) and the `nlp_data` directory (containing your PDFs) in the project's root directory.
2. Ensure all necessary files are available in these directories.

### Running the Application

1. Start the Flask API:
   ```bash
   python backend.py
   ```
   This will launch the API on `localhost:5003`.

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
   The interface will be available at `localhost:8501` on your local machine.

3. Optionally, explore the `demo.py` notebook for additional functionalities and demonstrations of the project's capabilities.

### RAG Evaluation

1. Create evaluation dataset using the code provided in `prepare_ragas_set.ipynb`.

2. Run the code in `run_ragas.py`.

3. The evaluation results will be saved in `mini_result.txt`.
