from typing import List
from src.interface.base_response_generator import BaseResponseGenerator
from src.util.invoke_ai import invoke_ai


SYSTEM_PROMPT = """
Eres un analista financiero experto en empresas argentinas.
Responde en español con base EXCLUSIVA en el contexto provisto.
Tu tarea es interpretar correctamente términos financieros y sus equivalentes abreviados
(por ejemplo: "EBITDA Aj." = "EBITDA Ajustado", "Flujo libre" = "FCF").
Entiende sinónimos y siglas, y prioriza los datos consolidados sobre los de segmentos.
Si hay números o valores explícitos, devuélvelos de forma clara y unificada.
Si el dato no está en el texto, razona con lo más cercano posible dentro del contexto, pero no inventes.
"""


class ResponseGenerator(BaseResponseGenerator):
    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate a response using OpenAI's chat completion."""
        # Combine context into a single string
        context_text = "\n".join(context)
        user_message = (
            f"<context>\n{context_text}\n</context>\n"
            f"<question>\n{query}\n</question>"
        )

        return invoke_ai(system_message=SYSTEM_PROMPT, user_message=user_message)