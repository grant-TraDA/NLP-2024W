import os

from transformers import pipeline
from openai import OpenAI
import torch

from rag.database import RAGSystem


class ArtExpertWithRAG:
    def __init__(
            self,
            rag_system: RAGSystem,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
            max_context_length: int = 2048,
            translation_model: str = "gpt-4o-mini"
    ):
        """
        Initialize the Art Expert.

        :param rag_system: Instance of RAGSystem for document search
        :param device: Device to run the model on ("cuda" or "cpu")
        :param model_name: Hugging Face model name
        :param max_context_length: Maximum context length in tokens
        :param translation_model: OpenAI model to use for translation
        """
        self.rag_system = rag_system
        self.device = device
        self.max_context_length = max_context_length
        self.translation_model = translation_model

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("Error: OPENAI_API_KEY environment variable not set")
            exit(1)

        # Initialize OpenAI client for translation
        self.translator = OpenAI(api_key=openai_api_key)

        # Initialize Llama 3.2 tokenizer and model
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=self.device,
            torch_dtype="auto",
        )

        self.base_system_prompt = """You are an art expert who is on an art exhibition organized by Academy of Fine Arts in Warsaw. Use the provided context to give accurate answers, 
        but don't directly reference these context fragments. If information isn't in the context, use your general knowledge.
        Always maintain a friendly and professional tone.

        Important rules:
        1. Provide clear and accurate information about art
        2. Maintain coherence and fluidity in responses
        3. Keep answers concise unless specifically asked for more detail
        4. If no context is provided, use your general knowledge
        5. Your name is Art Chat"""

    def _translate(self, text: str, target_lang: str) -> str:
        """Translate text between Polish and English using OpenAI."""
        direction = "Polish to English" if target_lang == "en" else "English to Polish"
        try:
            response = self.translator.chat.completions.create(
                model=self.translation_model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a professional translator. Translate the following text from {direction}. Maintain the original tone and meaning. The context is about art and Academy of Fine Arts in Warsaw."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return text  # Fallback to original text if translation fails

    def _prepare_context(self, english_query: str, num_results: int = 3) -> str:
        """Prepare context from RAG search results."""
        search_results = self.rag_system.search(
            query=english_query,
            top_k=num_results,
            include_metadata=True
        )

        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(
                f"{result['text']}\n"
            )
        return "\n".join(context_parts)

    def _format_conversation(
            self,
            context: str,
            english_query: str,
            conversation_history: list[dict] | None = None
    ) -> str:
        """
        Format the conversation for the model.

        :param context: RAG context in English
        :param english_query: Current user query in English
        :param conversation_history: Optional conversation history

        :return: Formatted conversation text
        """
        # Format system prompt with context
        formatted_text = f"<s>[INST] <<SYS>>\n{self.base_system_prompt}\n\nCONTEXT:\n{context}\n<</SYS>>\n\n"

        # Add translated conversation history
        if conversation_history:
            for message in conversation_history:
                content = message['content']
                if message["role"] == "user":
                    english_content = self._translate(content, "en")
                    formatted_text += f"{english_content} [/INST] "
                else:
                    formatted_text += f"{content} </s><s>[INST] "

        # Add current translated query
        formatted_text += f"{english_query} [/INST]"

        return formatted_text

    def get_response(
            self,
            user_query: str,
            conversation_history: list[dict] | None = None,
            temperature: float = 0.7,
            max_new_tokens: int = 1000
    ) -> str:
        """
        Get response from the art expert system with translation handling.

        :param user_query: User's question in Polish
        :param conversation_history: Optional list of previous messages
        :param temperature: OpenAI temperature parameter
        :param max_new_tokens: Maximum number of tokens to generate

        :return: Assistant's response in Polish
        """
        try:
            # Prepare context and format conversation (includes translation to English)
            eng_query = self._translate(user_query, "en")
            print(eng_query)
            context = self._prepare_context(eng_query)
            formatted_input = self._format_conversation(
                context,
                eng_query,
                conversation_history
            )
            print(context)
            print(formatted_input)

            # Generate response using the pipeline
            response = self.generator(
                formatted_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=1
            )[0]

            # Extract the generated text (removing the input prompt)
            english_response = response['generated_text'][len(formatted_input):].strip()

            # Translate response back to Polish
            polish_response = self._translate(english_response, "pl")

            return polish_response

        except Exception as e:
            error_msg = f"Przepraszamy, wystąpił błąd: {str(e)}"
            return error_msg
