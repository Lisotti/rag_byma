# from src.interface.base_evaluator import BaseEvaluator, EvaluationResult
# from src.util.invoke_ai import invoke_ai
# from src.util.extract_xml import extract_xml_tag

SYSTEM_PROMPT = """
You are a system that evaluates the correctness of a response to a question.
The question will be provided in <question>...</question> tags.
The response will be provided in <response>...</response> tags.
The expected answer will be provided in <expected_answer>...</expected_answer> tags.

The response doesn't have to exactly match all the words/context the expected answer. It just needs to be right about
the answer to the actual question itself.

Evaluate whether the response is correct or not, and return your reasoning in <reasoning>...</reasoning> tags.
Then return the result in <result>...</result> tags — either as 'true' or 'false'.
"""


# class Evaluator(BaseEvaluator):
#     def evaluate(
#         self, query: str, response: str, expected_answer: str
#     ) -> EvaluationResult:

#         user_prompt = f"""
#         <question>\n{query}\n</question>
#         <response>\n{response}\n</response>
#         <expected_answer>\n{expected_answer}\n</expected_answer>
#         """

#         response_content = invoke_ai(
#             system_message=SYSTEM_PROMPT, user_message=user_prompt
#         )

#         reasoning = extract_xml_tag(response_content, "reasoning")
#         result = extract_xml_tag(response_content, "result")
#         print(response_content)

#         if result is not None:
#             is_correct = result.lower() == "true"
#         else:
#             is_correct = False
#             reasoning = f"No result found: ({response_content})"

#         return EvaluationResult(
#             question=query,
#             response=response,
#             expected_answer=expected_answer,
#             is_correct=is_correct,
#             reasoning=reasoning,
#         )

import difflib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict
from dotenv import load_dotenv
from openai import OpenAI

from src.interface import BaseEvaluator, EvaluationResult


@dataclass
class Evaluator(BaseEvaluator):
    """
    Evalúa las respuestas generadas por el pipeline contra las esperadas.
    Usa comparación textual y, opcionalmente, embeddings semánticos.
    """

    def __post_init__(self):
        self.client = OpenAI()

    # --------------------------------------------------------------
    # 🔹 Método principal de evaluación
    # --------------------------------------------------------------
    def evaluate(self, question: str, response: str, expected_answer: str) -> EvaluationResult:
        """
        Compara una respuesta generada con la respuesta esperada.
        Retorna un EvaluationResult con razonamiento y score.
        """

        try:
            similarity = self._text_similarity(response, expected_answer)
            is_correct = similarity >= 0.75  # umbral configurable
            reasoning = f"Similarity: {similarity:.2f}. "

            # Si la similitud es baja, intentamos un chequeo semántico con embeddings
            if not is_correct:
                semantic_score = self._semantic_similarity(response, expected_answer)
                reasoning += f"Semantic Similarity: {semantic_score:.2f}. "
                is_correct = semantic_score >= 0.80

            if is_correct:
                reasoning += "✅ La respuesta coincide con la esperada o es semánticamente equivalente."
            else:
                reasoning += "❌ La respuesta no coincide ni semánticamente con la esperada."

        except Exception as e:
            reasoning = f"⚠️ Error en evaluación: {e}"
            is_correct = False

        return EvaluationResult(
            question=question,
            response=response,
            expected_answer=expected_answer,
            is_correct=is_correct,
            reasoning=reasoning,
        )

    # --------------------------------------------------------------
    # 🔹 Comparación literal / fuzzy
    # --------------------------------------------------------------
    def _text_similarity(self, text_a: str, text_b: str) -> float:
        """
        Usa difflib.SequenceMatcher para medir similitud literal entre textos.
        """
        ratio = difflib.SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()
        return round(ratio, 3)

    # --------------------------------------------------------------
    # 🔹 Comparación semántica (embeddings)
    # --------------------------------------------------------------
    def _semantic_similarity(self, text_a: str, text_b: str) -> float:
        """
        Usa embeddings de OpenAI para calcular similitud coseno entre textos.
        Requiere una API key válida.
        """
        try:
            # evita rate limit
            time.sleep(1.5)

            emb = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=[text_a, text_b],
            )

            vec_a, vec_b = emb.data[0].embedding, emb.data[1].embedding
            dot = sum(a * b for a, b in zip(vec_a, vec_b))
            norm_a = sum(a * a for a in vec_a) ** 0.5
            norm_b = sum(b * b for b in vec_b) ** 0.5
            cosine_sim = dot / (norm_a * norm_b)
            return round(cosine_sim, 3)

        except Exception as e:
            print(f"⚠️ Error calculando similitud semántica: {e}")
            return 0.0
