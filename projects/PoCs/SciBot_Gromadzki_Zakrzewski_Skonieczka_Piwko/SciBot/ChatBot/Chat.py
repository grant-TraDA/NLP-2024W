from typing import Dict

from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama


class SciBot:
    def __init__(self, llm: str) -> None:
        self.store = {}

        self.llm = ChatOllama(model=llm)
        # ===============================================

        ### Contextualize question ###
        self.contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        ### Answer question ###
        self.system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

    def ingest(self, db_path: str) -> None:
        """
        Load the database and create the conversational chain.
        """
        model = "hkunlp/instructor-xl"
        kwargs = {"device": "cpu"}
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=model,
            model_kwargs=kwargs,
        )

        db = FAISS.load_local(
            folder_path=db_path,
            index_name="faiss_index",
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

        self.retriever = db.as_retriever(
            search_type="mmr",  # “similarity” (default), “mmr”, or “similarity_score_threshold”
            search_kwargs={"k": 6},
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        self.rag_chain = create_retrieval_chain(
            history_aware_retriever, self.question_answer_chain
        )

        self.conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        """
        Get the chat history for a given session ID.
        """
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def ask(self, query: str, session_id: str = "abc123") -> Dict[str, str]:
        """
        Ask a question and get a response.
        """
        response = self.conversational_rag_chain.invoke(
            {"input": query},
            config={
                "configurable": {"session_id": session_id},
            },
        )
        return response
