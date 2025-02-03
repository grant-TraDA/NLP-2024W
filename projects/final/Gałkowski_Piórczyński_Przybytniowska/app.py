import streamlit as st

from modules.llm.generation import MODEL_NAME as LLM_MODEL_NAME
from modules.llm.generation import generate_response
from modules.llm.prompts import construct_llm_prompt
from modules.llm.retrieval import retrieve_k_most_similar_chunks

ICONS = {
    "bot": "🤖",
    "user": "👨",
}
CHUNK_SEPARATOR = "\n\n" + "####" * 20 + "\n\n"
GENERATION_KWARGS = {"temperature": 0.2, "min_p": 0.1}

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "bot",
            "content": "Cześć! Jestem chatbotem, który może pomóc Ci w informacjach dotyczących studiów na wydziale MiNI. Zadaj mi dowolne pytanie!",
        }
    ]

st.set_page_config(page_title="MiNI RAG Chatbot", page_icon="🤖")

st.title("MiNI RAG Chatbot")


for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=ICONS[message["role"]]):
        st.write(message["content"])

if user_input := st.chat_input(
    "Zadaj dowolne pytanie związane ze studiami na wydziale MiNI..."
):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar=ICONS["user"]):
        st.write(user_input)

    container_for_response = st.chat_message("bot", avatar=ICONS["bot"])

    with container_for_response:
        with st.spinner("Thinking..."):
            chunks = retrieve_k_most_similar_chunks(user_input, k=5)

            prompt = construct_llm_prompt(
                user_question=user_input,
                chunks=chunks,
                chat_history=st.session_state.messages,
            )

            response = generate_response(prompt, LLM_MODEL_NAME, **GENERATION_KWARGS)
            response = response["choices"][0]["message"]["content"]

    st.session_state.messages.append(
        {
            "role": "bot",
            "content": response,
        }
    )

    with container_for_response:
        st.write(response)
