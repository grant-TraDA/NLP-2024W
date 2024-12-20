import os
import ollama
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import TransformChain

import vector_database
import embedding
import wikipedia


class WikiRAG:
    def __init__(self, 
                 db_name: str = "wiki_db",
                 model_name: str = "llama3.1:8b",
                 documents_retrieved: int = 4):
        self.db = vector_database.VectorDatabaseWraper(db_name)
        self.model_name = model_name
        self.documents_retrieved = documents_retrieved

    def db_search(self, input_: str):
        emb = embedding.embedding([input_])[0]
        search_results, likeness = self.db.search(emb, self.documents_retrieved)
        return search_results, likeness

    def wikipedia_retriever(self, input_: dict) -> str:
            search_results, _ = self.db_search(input_)
            documents = []
            for result in search_results:
                section_text = wikipedia.get_section_text(result['page_title'], 
                                                          result['section_title'])
                documents.append(section_text)
            return "\n".join(documents)
    
    def query(self, question):
        template_RAG = "Try to answer the question based on YOUR knowledge \
            OR the context. Context: {context} \n Question: {question}"
        prompt = ChatPromptTemplate.from_template(template_RAG)
        model = ChatOllama(model = self.model_name)
        chain_wiki = (
            {"context": self.wikipedia_retriever, "question": RunnablePassthrough()}
            | prompt
            | model
        )
        return chain_wiki.invoke(question)
        
