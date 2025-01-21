import streamlit as st

from Chat import SciBot


def reset_conversation() -> None:
    st.session_state["messages"] = []
    st.session_state.context = None
    st.session_state["assistant"].store = {}


# ==========================================================================
# APP
# ==========================================================================

# Paths
db_path = "assets/db_instructor"
user_avatar = "assets/user.png"
bot_avatar = "assets/bot.png"

# Initial page config
st.set_page_config(page_title="SciBot", layout="wide")
st.title("SciBot")
footer_html = """<footer>
<p>NLP 2024</p>
</footer>"""
st.markdown(footer_html, unsafe_allow_html=True)


# Main body
def cs_body():

    # Initialization
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "assistant" not in st.session_state:
        st.session_state["assistant"] = SciBot(llm="qwen2.5:7b-instruct-q4_0")
        st.session_state["assistant"].ingest(db_path)

    if "is_disabled" not in st.session_state:
        st.session_state["is_disabled"] = False

    st.subheader("Chat")

    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message(message["role"], avatar=user_avatar):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"], avatar=bot_avatar):
                st.markdown(message["content"])

    # Process user input
    if (
        user_text := st.chat_input(
            "Write your query", disabled=st.session_state.is_disabled
        )
        or st.session_state.is_disabled
    ):

        if not st.session_state.is_disabled:
            st.session_state.messages.append(
                {"role": "user", "content": user_text, "avatar": user_avatar}
            )

            with st.chat_message("user", avatar=user_avatar):
                st.session_state.is_disabled = True
                st.markdown(user_text)
                st.rerun()

        with st.chat_message("assistant", avatar=bot_avatar):
            response_placeholder = st.empty()

            with st.spinner(f"Thinking..."):
                user_text = st.session_state.messages[-1]["content"]
                response = st.session_state["assistant"].ask(user_text)
                answer = response["answer"]

            st.session_state.is_disabled = False
            response_placeholder.markdown(answer)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "avatar": bot_avatar,
                }
            )
            st.rerun()

        st.button("Reset Chat", on_click=reset_conversation)


def main():
    cs_body()


if __name__ == "__main__":
    main()
