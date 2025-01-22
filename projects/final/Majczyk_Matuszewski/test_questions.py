import json
import yaml
import requests
import time
import psycopg2
from src.database_functions import (
    retrieve_related_articles,
    create_connection,
    retrieve_and_rerank,
)
from src.text_functions import format_context
from src.embedding_functions import get_embeddings
from tqdm import tqdm
from typing import List, Dict, Tuple
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_questions(file_path: str) -> List[Dict]:
    """Load questions from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_options(answers: List[Dict]) -> str:
    """Format answer options for the prompt."""
    return "\n".join(
        [f"{chr(65 + i)}. {answer['answer']}" for i, answer in enumerate(answers)]
    )


def create_prompt(question: str, answers: List[Dict], context: str = "") -> str:
    """Create formatted prompt in Polish."""
    options = format_options(answers)
    base_prompt = f"""Jesteś prawnikiem specjalizującym się w prawie konstytucyjnym. Odpowiedz na poniższe pytanie wybierając jedną z podanych odpowiedzi.
    
WAŻNE: W odpowiedzi podaj TYLKO literę odpowiedzi (A, B lub C), którą uważasz za prawidłową.

{f'Kontekst: {context}' if context else ''}

Pytanie: {question}

Odpowiedzi:
{options}

Odpowiedź:"""
    return base_prompt


def get_model_response(prompt: str, config: dict) -> str:
    """Get response from the Ollama API."""
    url = config["api"]["url"]
    payload = {"model": config["api"]["model"], "prompt": prompt, "stream": False}

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return ""
    except Exception as e:
        print(f"Error making API request: {e}")
        return ""


def evaluate_response(response: str, answers: List[Dict]) -> bool:
    """
    Evaluate if the model's response matches the correct answer.
    Expects response to be a single letter (A, B, or C).
    """
    # Find the index of the correct answer
    correct_index = next(i for i, ans in enumerate(answers) if ans["correct"])
    correct_letter = chr(65 + correct_index)  # Convert index to letter (A, B, C)

    # Clean the response and extract the first letter
    response = response.strip().upper()
    response_letter = response[0] if response else ""

    return response_letter == correct_letter


def run_evaluation(
    questions: List[Dict],
    config: dict,
    use_retrieval: bool = False,
    use_reranker: bool = False,
) -> List[Dict]:
    """
    Run evaluation for a specific configuration.
    """
    results = []
    connection = (
        create_connection(
            dbname="pgvector_nlp_db",
            user="postgres",
            password="yourpassword",
            host="localhost",
            port="5432",
        )
        if use_retrieval
        else None
    )

    for question in tqdm(questions, desc="Processing questions"):
        question_text = question["question"]
        answers = question["answers"]
        correct_answer = next(ans["answer"] for ans in answers if ans["correct"])
        print(f"Number of retrieved articles: {config['retrieval']['num_articles']}")

        context_section = ""
        if use_retrieval:
            try:
                if use_reranker:
                    related_articles = retrieve_and_rerank(
                        conn=connection,
                        text=question_text,
                        num_initial_candidates=config["retrieval"].get(
                            "num_initial_candidates", 20
                        ),
                        num_results=config["retrieval"]["num_articles"],
                    )
                else:
                    related_articles = retrieve_related_articles(
                        conn=connection,
                        text=question_text,
                        num_articles=config["retrieval"]["num_articles"],
                    )
                context_section = format_context(related_articles)
            except Exception as e:
                print(f"Error during retrieval: {e}")
                context_section = ""

        prompt = create_prompt(
            question=question_text, answers=answers, context=context_section
        )

        print(f"Question: {question_text}")
        print(f"Correct answer: {correct_answer}")
        print(f"Prompt:\n{prompt}\n")

        response = get_model_response(prompt, config)
        is_correct = evaluate_response(response, answers)

        print(f"Model response: {response}")
        print(f"Correct: {is_correct}\n")
        print("-" * 100)

        results.append(
            {
                "question": question_text,
                "correct_answer": correct_answer,
                "model_response": response,
                "is_correct": is_correct,
                "use_retrieval": use_retrieval,
                "use_reranker": use_reranker,
                "context_provided": bool(context_section),
            }
        )

    if connection:
        connection.close()

    return results


def main():
    # Load configuration and questions
    config = load_config()
    questions = load_questions(
        r"C:\Users\adamm\Documents\PW\Sem9\NLP\lAIwyer-NLP\data\constituion_questions.json"
    )

    # Run evaluations for all three configurations
    configurations = [
        # {"use_retrieval": False, "use_reranker": False, "name": "Model podstawowy"},
        {"use_retrieval": True, "use_reranker": False, "name": "Model z kontekstem"},
        # {
        #     "use_retrieval": True,
        #     "use_reranker": True,
        #     "name": "Model z kontekstem i rerankerem",
        # },
    ]

    all_results = []
    for conf in configurations:
        print(f"\nUruchamiam testy dla konfiguracji: {conf['name']}")
        results = run_evaluation(
            questions=questions,
            config=config,
            use_retrieval=conf["use_retrieval"],
            use_reranker=conf["use_reranker"],
        )
        all_results.extend(results)

    # Convert results to DataFrame for analysis
    df = pd.DataFrame(all_results)

    # Calculate and display metrics
    print("\nWyniki ewaluacji:")
    print("=" * 50)
    for conf in configurations:
        mask = (df["use_retrieval"] == conf["use_retrieval"]) & (
            df["use_reranker"] == conf["use_reranker"]
        )
        subset = df[mask]
        accuracy = subset["is_correct"].mean() * 100
        total = len(subset)
        correct = subset["is_correct"].sum()

        print(f"\n{conf['name']}:")
        print(f"Dokładność: {accuracy:.2f}%")
        print(f"Poprawne odpowiedzi: {correct}/{total}")

        if conf["use_retrieval"]:
            context_success = subset["context_provided"].mean() * 100
            print(f"Skuteczność pobierania kontekstu: {context_success:.2f}%")

    # Save detailed results to file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = f"evaluation_results_{timestamp}.csv"
    df.to_csv(results_file, index=False)
    print(f"\nSzczegółowe wyniki zapisano do: {results_file}")

    # Save configuration summary
    summary = {
        "timestamp": timestamp,
        "model": config["api"]["model"],
        "liczba_pytan": len(questions),
        "testowane_konfiguracje": [conf["name"] for conf in configurations],
        "ustawienia_pobierania": {
            "liczba_artykulow": config["retrieval"]["num_articles"],
            "liczba_kandydatow": config["retrieval"].get("num_initial_candidates", 20),
        },
    }

    summary_file = f"evaluation_summary_{timestamp}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Podsumowanie ewaluacji zapisano do: {summary_file}")


if __name__ == "__main__":
    main()
