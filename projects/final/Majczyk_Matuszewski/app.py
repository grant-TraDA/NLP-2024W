import streamlit as st
from streamlit_extras.tags import tagger_component
import requests
import json
import yaml
import time
from src.database_functions import retrieve_related_articles, create_connection, retrieve_and_rerank
from src.text_functions import format_context, custom_print

# Load configuration from YAML file
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

st.set_page_config(
    page_title=config["app"]["title"], page_icon=config["app"]["page_icon"]
)

TEMPLATE = config["template"]

st.title(config["app"]["title"])
tab1, tab2 = st.tabs(config["app"]["tabs"])

# Load personalities
with open(config["personalities_file"], "r", encoding="utf-8") as f:
    personalities = json.load(f)

with tab1:

    col1, col2 = st.columns([3, 1])
    
    with col1:
        if "personality" not in st.session_state:
            st.session_state.personality = list(personalities.keys())[0]

        system_instruction = personalities[st.session_state.personality]

        st.selectbox(
            "Wybierz osobowość:",
            list(personalities.keys()),
            index=list(personalities.keys()).index(st.session_state.personality),
            key="personality",
        )
    with col2:
        use_reranker = st.checkbox(
            'Użyj reranker', value=False, 
            help='Włącz, aby użyć zaawansowanego rankingu wyników'
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    def response_generator():
        connection = create_connection(
            port=config["database"]["port"],
        )

        if use_reranker:
            related_articles = retrieve_and_rerank(
                conn=connection,
                text=prompt,
                num_initial_candidates=config["retrieval"].get("num_initial_candidates", 20),
                num_results=config["retrieval"]["num_articles"]
            )
        else:
            related_articles = retrieve_related_articles(
                conn=connection,
                text=prompt,
                num_articles=config["retrieval"]["num_articles"],
            )
    
        context_section = format_context(related_articles)
        formatted_prompt = TEMPLATE.format(
            system_instruction=system_instruction,
            context_section=context_section,
            user_query=prompt,
        )
        print(formatted_prompt)
        print("-" * 100, end="\n\n")
        # full_response = (
        #     f"Jestem {personalities[st.session_state.personality]}" + formatted_prompt
        # )
        url = config["api"]["url"]
        payload = {
            "model": config["api"]["model"],
            "prompt": formatted_prompt,
        }

        response = requests.post(
            url, json=payload, stream=config["response_generation"]["stream"]
        )

        full_response = ""

        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        full_response += data.get("response", "")
                    except json.JSONDecodeError as e:
                        print("Error decoding JSON line:", line.decode("utf-8"))
        else:
            print(f"Error: {response.status_code}, {response.text}")
            full_response = f"Coś poszło nie tak. Spróbuj ponownie. Jak coś to jestem {personalities[st.session_state.personality]}."
        for word in full_response.split():
            yield word + " "
            time.sleep(config["response_generation"]["delay_per_word"])

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"👋 Zadaj mi pytanie o Konstytucję RP."):
        # Append the user's message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display the user's message in the chat
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and append the assistant's response
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator())
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

with tab2:
    st.header(config["app"]["tabs"][1])
    # List of contacts
    if "contact" in config:
        for name, url in config["contact"].items():
            st.markdown(f"- **{name.capitalize()}**: [link]({url})")
