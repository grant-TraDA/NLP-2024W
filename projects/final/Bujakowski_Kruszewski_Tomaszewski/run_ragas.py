from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
import json
from ragas.run_config import RunConfig


llm = ChatOllama(model = "llama3")
embeddings = OllamaEmbeddings(model = "llama3")

with open('mini_dataset.json', 'r') as fin:
    dataset = json.load(fin)

from ragas import EvaluationDataset
eval_dataset = EvaluationDataset.from_list(dataset)

from ragas import evaluate
from ragas.metrics import context_precision, faithfulness, answer_relevancy, context_recall

result = evaluate(eval_dataset, metrics=[context_precision, faithfulness, answer_relevancy, context_recall], llm=llm, embeddings=embeddings,run_config=RunConfig(max_workers=1))

print(result)

with open('mini_result.txt', 'w') as fout: 
    fout.write(f"Context Precision: {result['context_precision']}\n")
    fout.write(f"Faithfulness: {result['faithfulness']}\n")
    fout.write(f"Answer Relevancy: {result['answer_relevancy']}\n")
    fout.write(f"Context Recall: {result['context_recall']}\n")