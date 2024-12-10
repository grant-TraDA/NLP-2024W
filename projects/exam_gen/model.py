from typing import List, Optional

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, model_validator

question_answer_template = ChatPromptTemplate([
    ("system", "Based on the following text, answer user's question, make the answer short and preceise while sticking to the context: {context}"),
    ("user", "{question}")
]
)
closed_question_answer_template = ChatPromptTemplate([
    ("system", "Based on the following text, answer user's question, make the answer short and preceise while sticking to the context: {context}"),
    ("user", "{question} \n\n Possible choices: {choices}")
]
)

answer_validation_template = ChatPromptTemplate([
    ("system", "As an teacher, you are asked to validate the answer to the following question on a scale from 1 (completely incorrect) to 5 (excelent) The question is: {question}. The context is: {context}. Example correct answers with grade 5: {correct_answer}. Is this answer correct?"),
    ("user", "{answer}"),
    ("system", "Remember to output only the grade of the answer from 1 to 5 ONLY!")
]
)
question_generation_template = ChatPromptTemplate([
    ("system", "Based on the context given by user, generate a question that can be answered using the mentioned text, remember that the question will be answered without looking at that context, so generate the question which will allow studnets faimiar with it to answer it correctly."),
    ("user", "{context}")
]
)

class OpenEndedQuestion(BaseModel):
    question: str = Field(..., description="The question text.")
    example_correct_answers: List[str] = Field(
        ...,
        description="Examples of correct answers to the question."
    )
    context: str = Field(
        ...,
        description="The context in which the question is being asked."
    )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        questions = []
        for _, row in df.iterrows():
            # row['answers']['text'] is typically a list of correct answers in SQuAD
            questions.append(cls(
                question=row['question'],
                example_correct_answers=row['answers']['text'],
                context=row['context']
            ))
        return questions

    def generate_question_from_context(self, llm):
        messages = question_generation_template.format_messages(context=self.context)
        response = llm.invoke(messages)
        return response.content.strip()

    def generate_llm_response(self, llm: BaseChatModel):
        messages = question_answer_template.format_messages(
            question=self.question, context=self.context
        )
        response = llm.invoke(messages)
        return response.content.strip()

    def validate_answer(self, answer: str, llm: BaseChatModel):
        messages = answer_validation_template.format_messages(
            question=self.question,
            context=self.context,
            correct_answer=self.example_correct_answers,
            answer=answer
        )
        response = llm.invoke(messages).content.strip()

        # parse the numeric score
        try:
            grade = int(response)
            if grade < 1 or grade > 5:
                grade = 0
        except ValueError:
            grade = 0
        return grade


class CloseEndedQuestionTemplate(BaseModel):
    question: str = Field(..., description="The question text.")
    choices: list[str] = Field(
        ...,
        description="List of answer choices in which only one is correct, and the rest are incorrect."
    )
    correct_answer: int = Field(
        ...,
        description="Index of the correct answer in the choices list. Indexing starts from 0.",
        ge=0,  # Ensure the index is non-negative
    )

    @model_validator(mode='after')
    def check_correct_answer_in_range(self):
        if self.correct_answer < 0 or self.correct_answer >= len(self.choices):
            raise ValueError("correct_answer must be within the range of choices indices")
        return self


class LLMAnswerQuestionTemplate(BaseModel):
    index: int = Field(..., description="Index of the correct answer in the choices list. Indexing starts from 0.")


class CloseEndedQuestion(BaseModel):
    context: str = Field(
        ...,
        description="The context in which the question is being asked."
    )
    question: Optional[CloseEndedQuestionTemplate] = Field(
        None,
        description="The parsed question object."
    )

    def generate_question_from_context(self, llm: BaseChatModel) -> CloseEndedQuestionTemplate:
        messages = question_generation_template.format_messages(context=self.context)
        llm_with_structured_output = llm.with_structured_output(CloseEndedQuestionTemplate)
        response = llm_with_structured_output.invoke(messages)
        return response

    def generate_llm_response(self, llm: BaseChatModel) -> LLMAnswerQuestionTemplate:
        if self.question is None:
            raise ValueError("Question template must be set before generating LLM response.")
        messages = closed_question_answer_template.format_messages(
            question=self.question.question,
            context=self.context,
            choices=self.question.choices
        )
        llm_with_structured_output = llm.with_structured_output(LLMAnswerQuestionTemplate)
        response = llm_with_structured_output.invoke(messages)
        return response

    def validate_answer(self, answer: int) -> int:
        if self.question is None:
            raise ValueError("Question template must be set before generating LLM response.")

        if answer < 0 or answer >= len(self.question.choices):
            raise ValueError("Answer index must be within the range of choices indices")

        if answer not in self.choices:
            return 0  # invalid choice

        # Check if the provided answer matches the correct answer
        if answer == self.question.correct_answer:
            return 5
        else:
            return 1

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        questions = []
        for _, row in df.iterrows():
            # row['answers']['text'] is typically a list of correct answers in SQuAD
            questions.append(cls(
                context=row['context']
            ))

        return questions
