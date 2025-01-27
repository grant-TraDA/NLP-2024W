import streamlit as st

from modules.llm.generation import generate_response
from modules.llm.prompts import get_prompt
from modules.llm.retrieval import retrieve_k_most_similar_chunks

ICONS = {
    "bot": "🤖",
    "user": "👨",
}
CHUNK_SEPARATOR = "\n\n" + "####" * 20 + "\n\n"

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
            joined_chunks = (
                "####" * 20 + "\n\n" + CHUNK_SEPARATOR.join(chunks) + "\n\n" + "####" * 20
            )

            prompt = get_prompt().format(
                user_question=user_input,
                chunks=joined_chunks,
            )

            response = generate_response(prompt)
            response = response["choices"][0]["message"]["content"]

        # response = "Przepraszam, nie potrafię odpowiedzieć na to pytanie. 😔"

    st.session_state.messages.append(
        {
            "role": "bot",
            "content": response,
        }
    )

    with container_for_response:
        st.write(response)
