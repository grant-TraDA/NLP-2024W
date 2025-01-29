# ChatBot evaluation

Evaluation was split into two parts - embedding evaluation and LLM evaluation.

## Embedding Evaluation

For the evaluation, we compared the four embedding models across the following steps:

1. **Model Selection:**
   We used the following embedding models for the evaluation:
   - Ollama - `snowflake-arctic-embed`
   - Ollama - `mxbai-embed-large`
   - Ollama - `nomic-embed-text`
   - HuggingFace - `instructor-xl`

2. **Embedding Generation:**
   The document chunks were embedded using each of these models, and vector stores were created using FAISS for fast indexing and similarity search. The resulting vector stores were distinct in terms of embedding values, but they all contained the same document chunks. This embedding generation step is performed in `embeddings/create_vdbs.ipynb`. The vector stores are saved to `data/dbs`.

3. **Query Generation:**
   To generate test queries, we randomly sampled 100 document chunks from the vector stores and used the `llama3.1:8b` model to rephrase their content. This created altered fragments of the documents to serve as input queries for testing the vector store’s accuracy. We saved metadata for each modified chunk, which included its index and context within the original document. The query generation is performed in `embeddings/gen_mod_documents.ipynb`. The modified document chunks are saved to `data/mod/`.

5. **Retrieval and Ranking:**
   After embedding the modified queries using each model, we retrieved the results from the respective vector stores. The quality of the model's retrieval was evaluated by checking the rank of the original chunk in the results—if the original chunk appeared higher in the ranking, the model was deemed more effective at retrieving relevant information. This retrieval and ranking evaluation is done in `embeddings/evaluate.ipynb`.

### Evaluation Metrics

We evaluated the performance of each embedding model using the following four metrics:

- **Hit Rate (HR@4):** Measures whether the correct document is included in the retrieved results.
- **Mean Reciprocal Rank (MRR):** Assesses the rank of the correct document, rewarding higher placements.
- **Normalized Discounted Cumulative Gain (nDCG):** Evaluates the quality of rankings, considering both the position and relevance of retrieved documents.
- **Mean Average Precision (MAP):** Measures precision at each rank where a correct document is retrieved, averaged across all queries.

All results were saved to `data/results/embeddings.csv`.

## LLM Evaluation

The second part of our evaluation process focused on the **SciBot** chatbot, which uses a vector store based on the `snowflake-arctic-embed` model. We evaluated several large language models integrated with the chatbot:

- `qwen2.5:3b`
- `qwen2.5:7b-instruct-q4_0`
- `llama3.1:3b`
- `llama3.2:8b`

### Evaluation Setup

1. **Question Set Creation:**  
   We created a set of 60 diverse questions targeting specific topics covered in the scientific papers within our database. These questions ranged from inquiries about particular methods described in the papers to general concepts, advantages, and disadvantages of the approaches discussed. The complete list of questions is available in `data/llm_eval/questions.txt`.

2. **Response Generation and Data Collection:**  
   For each question, the chatbot generated responses based on the `snowflake-arctic-embed` vector store and the LLM models mentioned above. The response, response time, and the retrieved context (six paper chunks) were saved. The response generation is handled in `chatbot/gen_responses.ipynb`. The responses are saved in `data/llm_eval/real_outputs.csv`.

3. **Ground Truth Generation:**  
   Ground truth answers for the 60 questions were generated using the `GPT-4o-mini` model. The model was provided with the questions and their corresponding retrieved contexts (six paper chunks for each question). This process ensured that the answers were based solely on the retrieved contexts, mitigating any knowledge advantage of the GPT model. The ground truth generation is done in `chatbot/gen_ground_truth.ipynb`. The ground truths are saved in `data/llm_eval/real_outputs_with_gt.csv`.

4. **Evaluation with `deepeval`:**  
   The responses were evaluated using the `deepeval` package, which employs the `GPT-4o-mini` model as the evaluator. This evaluator scores the responses based on:
   - **Answer Relevancy Metric:** Evaluates how well the response aligns with the question.
   - **Faithfulness Metric:** Measures factual consistency with the provided context.
   - **Contextual Precision:** Assesses the proportion of relevant information in the response.
   - **Contextual Recall:** Evaluates the chatbot's ability to extract relevant information from the context.
   - **Contextual Relevancy:** Combines precision and recall to assess overall contextual alignment.

   The evaluation process is performed in `chatbot/evaluate_chatbot.ipynb`.

All results were saved to `data/results/llm_metrics.csv`.

## Note

Additional packages are needed for evaluation - ```pip install -r evaluation/requirements.txt```
