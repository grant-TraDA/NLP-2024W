import pathlib
import os

current_path = pathlib.Path(__file__).parent.absolute()
pdf_path = os.path.join(current_path, "nlp_data")
logs_path = os.path.join(current_path, "logs")

import datetime
import uuid

import requests
import streamlit as st
from pdf2image import convert_from_bytes
from streamlit_text_rating.st_text_rater import st_text_rater


if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4()
if "query" not in st.session_state:
    st.session_state["query"] = ""
if "show_clear" not in st.session_state:
    st.session_state["show_clear"] = False
if "img_generated" not in st.session_state:
    st.session_state["img_generated"] = False
if "likeddisliked" not in st.session_state:
    st.session_state["likeddisliked"] = None

st.set_page_config(
    layout="wide",
    page_title="MiNI database",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""",
    unsafe_allow_html=True,
)

if "file_names" not in st.session_state:
    st.session_state["file_names"] = []
if "page_labels" not in st.session_state:
    st.session_state["page_labels"] = []
if "excerpts" not in st.session_state:
    st.session_state["excerpts"] = []
if "scores" not in st.session_state:
    st.session_state["scores"] = []
if "images" not in st.session_state:
    st.session_state["images"] = []


def generate_answer(question, url="http://localhost:5003/ask"):
    payload = {
        "question": question,
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers)
    except:
        return "Error retrieving answer", "No source"
    if response.status_code == 200:
        data = response.json()
        return data.get("answer"), data.get("source")
    return "Error", "No source"


def rate_answer(rating):
    print(rating)
    with open(f"{logs_path}/ratings.log", "a") as f:
        f.write(
            f"{st.session_state['session_id']};{st.session_state['question']};{rating};{datetime.datetime.now().__str__()}\n"
        )


def comment_answer(comment):
    print(comment)
    with open(f"{logs_path}/comments.log", "a") as f:
        f.write(
            f"{st.session_state['session_id']};{st.session_state['question']};{comment};{datetime.datetime.now().__str__()}\n"
        )


def main():
    st.title("MiNI database")
    question = st.session_state.get("question", "")
    question = st.text_area(
        "Enter text:",
        value=question,
        placeholder="What do you want to know?",
        key="input_text",
    )
    st.session_state["question"] = question

    col1, col2, _ = st.columns([1, 1, 13])

    if not st.session_state.show_clear:
        main_button = col1.empty()
        submit_button = main_button.button("Submit")
        if submit_button:
            if question == "":
                return
            main_button.empty()
            st.session_state.show_clear = True
            with st.spinner("Generating results..."):
                try:
                    result, sources = generate_answer(question)
                    st.session_state["result"] = result
                    st.session_state["sources"] = sources
                except:
                    st.markdown("# Model not responding")
            st.rerun()
    if st.session_state.show_clear:
        clear_button = col1.button("Clear")
        if clear_button:
            st.session_state["show_clear"] = False
            st.session_state["question"] = ""
            st.session_state["likeddisliked"] = None
            if "result" in st.session_state:
                del st.session_state["result"]
            st.session_state["file_names"] = []
            st.session_state["page_labels"] = []
            st.session_state["excerpts"] = []
            st.session_state["scores"] = []
            st.session_state["images"] = []
            st.session_state["img_generated"] = False
            st.rerun()

    if (
        st.session_state.show_clear
        and "result" in st.session_state
        and st.session_state["result"] != "error"
    ):
        col1, _, col2 = st.columns([10, 1, 10])
        container1 = col1.container(height=600)
        container1.markdown("## Answer:")
        container1.markdown(
            f"<div align='justify'>{st.session_state['result']}</div>",
            unsafe_allow_html=True,
        )
        rating = st_text_rater("Rate the answer:", key="answer_rating")

        if rating != st.session_state["likeddisliked"]:
            print(
                f"Updating session state 'likeddisliked' from {st.session_state['likeddisliked']} to {rating}."
            )
            st.session_state["likeddisliked"] = rating
            print("New rating:", rating)
            rate_answer(rating)
        else:
            print("Rating has not changed.")

        comment = st.text_area("Comment on the answer:", value="", key="comment")
        submit_comment = st.button("Submit comment", key="submit_comment")
        if submit_comment:
            comment_answer(comment)

        with col2:
            if not st.session_state["img_generated"]:
                with st.spinner("Loading sources..."):
                    for i, source in enumerate(st.session_state["sources"]):
                        file_name = source["file_name"]
                        st.session_state["file_names"].append(file_name)

                        # Get page label
                        try:
                            page_label = source["page_label"]
                        except KeyError:
                            page_label = source["slide_num"]
                        st.session_state["page_labels"].append(page_label)

                        # Get other data
                        st.session_state["excerpts"].append(source["excerpt"])
                        st.session_state["scores"].append(source["score"])

                        if not "img" in source:
                            if file_name.endswith(".pdf"):
                                with open(
                                    f"{pdf_path}/{file_name}",
                                    "rb",
                                ) as f:
                                    pdf_bytes = f.read()
                                images_from_pdf = convert_from_bytes(
                                    pdf_bytes,
                                    single_file=True,
                                    first_page=int(page_label),
                                    last_page=int(page_label),
                                )
                                st.session_state["images"].append(images_from_pdf[0])
                            else:
                                st.session_state["images"].append(None)
                        else:
                            st.session_state["images"].append(source["img"])
                    st.session_state["img_generated"] = True
            with st.spinner("Printing sources..."):
                container2 = st.container(height=600)

                for i in range(len(st.session_state["file_names"])):
                    container2.markdown(
                        f"#### *{i+1}. {st.session_state['file_names'][i]}*, page - {st.session_state['page_labels'][i]}, (score - {st.session_state['scores'][i]})"
                    )
                    with container2.expander("Excerpt"):
                        st.markdown(
                            f"<div align='justify'>{st.session_state['excerpts'][i]}</div>",
                            unsafe_allow_html=True,
                        )

                    if (
                        st.session_state["file_names"][i].endswith(".pdf")
                        and st.session_state["images"][i] is not None
                    ):
                        with container2.expander("Source preview"):
                            try:
                                file_path = (
                                    f"{pdf_path}/{st.session_state['file_names'][i]}"
                                )
                                with open(file_path, "rb") as f:
                                    file_bytes = f.read()
                                st.image(
                                    st.session_state["images"][i],
                                    use_container_width=True,
                                )
                                st.download_button(
                                    label="Download Source",
                                    data=file_bytes,
                                    file_name=st.session_state["file_names"][i],
                                    key=f"downloadbutton{i}",
                                )
                            except Exception as e2:
                                print(e2)
    elif "result" in st.session_state and (
        st.session_state["result"] == "Error"
        or st.session_state["result"] == "Error retrieving answer"
    ):
        col1.markdown("# Error retrieving answer.")


if __name__ == "__main__":
    main()
