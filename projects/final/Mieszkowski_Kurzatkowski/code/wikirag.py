import os
import ollama
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import TransformChain

import vector_database
import embedding
import wiki


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
            print("##### DICTIONARY #####")
            print(search_results)
            documents = []
            for result in search_results:
                section_text = wiki.get_section_text(result['page_title'], 
                                                    result['section_title'],
                                                    result['subsection_title'],
                                                    result['subsubsection_title'],
                                                    result['part'])
                documents.append(section_text)

            print("##### Retrievals #####")
            retrievals = "\n".join(documents)
            print(retrievals)
            return retrievals
    
    def query(self, question):
        template_RAG = "You are given a context describing events in 2023 and 2024. Answer the question based on the context. If the context doesn't invlove reliable information say so. Context: {context} \n Question: {question}"
        prompt = ChatPromptTemplate.from_template(template_RAG)
        model = ChatOllama(model = self.model_name)
        chain_wiki = (
            {"context": self.wikipedia_retriever, "question": RunnablePassthrough()}
            | prompt
            | model
        )
        print("##### ANSWER #####")
        return chain_wiki.invoke(question)
