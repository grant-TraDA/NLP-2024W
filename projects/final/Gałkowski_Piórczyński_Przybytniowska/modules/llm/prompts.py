from typing import List

CHUNK_SEPARATOR = "\n\n" + "####" * 20 + "\n\n"


def get_system_prompt() -> str:
    return """
    Jesteś pomocnym asystentem AI odpowiedzialnym za pomoc studentom w ich pytaniach związanych ze studiowaniem na wydziale MiNI (Matematyki i Nauk Informacyjnych) i Politechnice Warszawskiej. Odpowiedz na pytanie użytkownika na podstawie pobranych fragmentów, a jeśli nie jesteś w stanie tego zrobić, poinformuj, że przy obecnej wiedzy nie jesteś w stanie odpowiedzieć na pytanie.

    """


def transform_history(chat_history: List[dict[str, str]]) -> str:
    llm_history = []

    for current_msg, next_msg in zip(chat_history, chat_history[1:]):
        if current_msg["role"] == "user" and next_msg["role"] == "bot":
            llm_history.append({"role": "user", "content": current_msg["content"]})
            llm_history.append({"role": "system", "content": next_msg["content"]})
    return llm_history


def construct_llm_prompt(
    user_question: str, chunks: List[str], chat_history: List[dict[str, str]]
) -> str:
    joined_chunks = (
        "####" * 20 + "\n\n" + CHUNK_SEPARATOR.join(chunks) + "\n\n" + "####" * 20
    )

    llm_prompt = [{"role": "system", "content": get_system_prompt()}]

    llm_prompt += transform_history(chat_history)

    context_and_user_query = f"""
    Pomocnicza wiedza:
    {joined_chunks}

    ------------------------------------------------------------------------------------

    Pytanie użytkownika:
    {user_question}

    ------------------------------------------------------------------------------------
    """

    llm_prompt.append({"role": "user", "content": context_and_user_query})

    return llm_prompt
