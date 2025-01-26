from typing import List, Dict, Optional
from openai import OpenAI
import json
from rag.database import PolishRAGSystem


class PolishArtExpertRAG:
    def __init__(
            self,
            rag_system: PolishRAGSystem,
            openai_api_key: str,
            model: str = "gpt-4o-mini",
            max_context_length: int = 20000
    ):
        """
        Initialize the Polish Art Expert RAG system.

        Args:
            rag_system: Instance of PolishRAGSystem for document search
            openai_api_key: OpenAI API key
            model: OpenAI model to use
            max_context_length: Maximum context length in tokens
        """
        self.rag_system = rag_system
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.max_context_length = max_context_length

        self.base_system_prompt = """Jesteś ekspertem w dziedzinie sztuki, który zawsze odpowiada w języku polskim. 
        Korzystaj z dostarczonych fragmentów kontekstu, aby udzielić dokładnej odpowiedzi, lecz nie odwołuj się bezpośrednio do tych fragmentów.
        Jeśli informacja nie znajduje się w kontekście, bazuj na swojej wiedzy ogólnej.
        Zawsze zachowuj przyjazny i profesjonalny ton wypowiedzi.

        Ważne zasady:
        1. Odpowiadaj tylko po polsku
        2. Zachowaj spójność i płynność wypowiedzi
        3. Udzielaj zwięzłych odpowiedzi, opowiadaj dłużej tylko jeśli zostaniesz bezpośrednio o to poproszony
        4. Nazywasz się Art Chat"""

    def _prepare_context(self, query: str, num_results: int = 3) -> str:
        """
        Prepare context from RAG search results.

        Args:
            query: User query
            num_results: Number of results to include

        Returns:
            Formatted context string
        """
        search_results = self.rag_system.search(
            query=query,
            top_k=num_results,
            include_metadata=True
        )

        context_parts = []
        for i, result in enumerate(search_results, 1):
            source_info = result.get('metadata', {}).get('filename', 'Brak źródła')
            context_parts.append(
                f"Fragment {i} (Źródło: {source_info}):\n{result['text']}\n"
            )
        print(context_parts)
        return "\n".join(context_parts)

    def get_response(
            self,
            user_query: str,
            conversation_history: Optional[List[Dict]] = None,
            temperature: float = 0.7
    ) -> str:
        """
        Get response from the art expert system.

        Args:
            user_query: User's question in Polish
            conversation_history: Optional list of previous messages
            temperature: OpenAI temperature parameter

        Returns:
            Assistant's response in Polish
        """
        # Prepare context from RAG
        context = self._prepare_context(user_query)

        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": f"{self.base_system_prompt}\n\nKONTEKST:\n{context}"
            }
        ]

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)

        # Add current user query
        messages.append({
            "role": "user",
            "content": user_query
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            error_msg = f"Przepraszamy, wystąpił błąd: {str(e)}"
            return error_msg
