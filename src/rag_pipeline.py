# from concurrent.futures import ThreadPoolExecutor
# from dataclasses import dataclass
# from typing import Dict, List, Optional
# from src.interface import (
#     BaseDatastore,
#     BaseIndexer,
#     BaseRetriever,
#     BaseResponseGenerator,
#     BaseEvaluator,
#     EvaluationResult,
# )


# @dataclass
# class RAGPipeline:
#     """Main RAG pipeline that orchestrates all components."""

#     datastore: BaseDatastore
#     indexer: BaseIndexer
#     retriever: BaseRetriever
#     response_generator: BaseResponseGenerator
#     evaluator: Optional[BaseEvaluator] = None

#     def reset(self) -> None:
#         """Reset the datastore."""
#         self.datastore.reset()

#     def add_documents(self, documents: List[str]) -> None:
#         """Index a list of documents."""
#         items = self.indexer.index(documents)
#         self.datastore.add_items(items)
#         print(f"✅ Added {len(items)} items to the datastore.")

#     def process_query(self, query: str) -> str:
#         search_results = self.retriever.search(query)
#         print(f"✅ Found {len(search_results)} results for query: {query}\n")

#         for i, result in enumerate(search_results):
#             print(f"🔍 Result {i+1}: {result}\n")

#         response = self.response_generator.generate_response(query, search_results)
#         return response

#     def evaluate(
#         self, sample_questions: List[Dict[str, str]]
#     ) -> List[EvaluationResult]:
#         # Evaluate a list of question/answer pairs.
#         questions = [item["question"] for item in sample_questions]
#         expected_answers = [item["answer"] for item in sample_questions]

#         with ThreadPoolExecutor(max_workers=10) as executor:
#             results: List[EvaluationResult] = list(
#                 executor.map(
#                     self._evaluate_single_question,
#                     questions,
#                     expected_answers,
#                 )
#             )

#         for i, result in enumerate(results):
#             result_emoji = "✅" if result.is_correct else "❌"
#             print(f"{result_emoji} Q {i+1}: {result.question}: \n")
#             print(f"Response: {result.response}\n")
#             print(f"Expected Answer: {result.expected_answer}\n")
#             print(f"Reasoning: {result.reasoning}\n")
#             print("--------------------------------")

#         number_correct = sum(result.is_correct for result in results)
#         print(f"✨ Total Score: {number_correct}/{len(results)}")
#         return results

#     def _evaluate_single_question(
#         self, question: str, expected_answer: str
#     ) -> EvaluationResult:
#         # Evaluate a single question/answer pair.
#         response = self.process_query(question)
#         return self.evaluator.evaluate(question, response, expected_answer)

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional
import re
import json
import time
from src.interface import (
    BaseDatastore,
    BaseIndexer,
    BaseRetriever,
    BaseResponseGenerator,
    BaseEvaluator,
    EvaluationResult,
)

# ======================================================
# 🧱 Helper Functions
# ======================================================

def ethical_guardrails(response_text: str) -> str:
    """
    Aplica reglas éticas y regulatorias antes de devolver la respuesta.
    """
    lowered = response_text.lower()
    if any(x in lowered for x in ["comprar", "vender", "inversión", "rentabilidad futura"]):
        return "⚠️ No se permiten recomendaciones de inversión ni proyecciones financieras."
    if any(x in lowered for x in ["predicción", "proyección", "estimado"]):
        return "⚠️ El sistema no realiza estimaciones ni predicciones. Solo resume información factual."
    return response_text + "\n\n📘 Fuente: Documentos IFRS públicos o reportes oficiales."


def is_grounded_in_context(response: str, context_list: List[str]) -> bool:
    """
    Verifica si la respuesta hace referencia explícita o semántica a los contextos recuperados.
    """
    context_text = " ".join(context_list).lower()
    resp = response.lower()
    # Si la respuesta contiene valores o frases exactas del contexto, se considera grounded
    match_ratio = len(set(resp.split()) & set(context_text.split())) / max(1, len(resp.split()))
    return match_ratio > 0.15  # umbral ajustable


def compute_metrics(results: List[EvaluationResult]) -> Dict[str, float]:
    """
    Calcula métricas globales del desempeño del sistema.
    """
    total = len(results)
    correct = sum(r.is_correct for r in results)
    grounded = sum(is_grounded_in_context(r.response, [r.expected_answer]) for r in results)
    hallucinated = sum(
        1 for r in results if not is_grounded_in_context(r.response, [r.expected_answer])
    )

    return {
        "accuracy": round(correct / total, 3),
        "groundedness": round(grounded / total, 3),
        "hallucination_rate": round(hallucinated / total, 3),
    }


# ======================================================
# 🚀 Main Pipeline
# ======================================================

@dataclass
class RAGPipeline:
    """Main RAG pipeline that orchestrates all components."""

    datastore: BaseDatastore
    indexer: BaseIndexer
    retriever: BaseRetriever
    response_generator: BaseResponseGenerator
    evaluator: Optional[BaseEvaluator] = None

    # ---------------------------------------------
    # Core Functions
    # ---------------------------------------------
    def reset(self) -> None:
        """Reset the datastore."""
        self.datastore.reset()

    def add_documents(self, documents: List[str]) -> None:
        """Index a list of documents."""
        items = self.indexer.index(documents)
        self.datastore.add_items(items)
        print(f"✅ Added {len(items)} items to the datastore.")

    def process_query(self, query: str) -> str:
        """Run the full RAG retrieval + generation pipeline."""
        search_results = self.retriever.search(query)
        print(f"✅ Found {len(search_results)} results for query: {query}\n")

        for i, result in enumerate(search_results):
            print(f"🔍 Result {i+1}: {result[:500]}...\n")

        # Generar respuesta base
        response = self.response_generator.generate_response(query, search_results)
        # Aplicar guardrails éticos
        response = ethical_guardrails(response)
        return response

    # ---------------------------------------------
    # Evaluación extendida
    # ---------------------------------------------
    def evaluate(self, sample_questions: List[Dict[str, str]]) -> List[EvaluationResult]:
        """
        Evalúa una lista de preguntas/respuestas esperadas,
        midiendo exactitud, groundedness y alucinación.
        """
        questions = [item["question"] for item in sample_questions]
        expected_answers = [item["answer"] for item in sample_questions]

        print(f"🧠 Starting evaluation with {len(questions)} questions...")
        results: List[EvaluationResult] = []

        for q, expected in zip(questions, expected_answers):
            time.sleep(6)  # evitar rate limit (Cohere u OpenAI)
            r = self._evaluate_single_question(q, expected)
            results.append(r)

        # Mostrar resultados uno por uno
        for i, result in enumerate(results):
            emoji = "✅" if result.is_correct else "❌"
            print(f"{emoji} Q{i+1}: {result.question}")
            print(f"Response: {result.response}\nExpected: {result.expected_answer}\n")
            print(f"Reasoning: {result.reasoning}\n{'-'*50}")

        # Cálculo de métricas globales
        metrics = compute_metrics(results)
        print("\n📊 Evaluation Summary:")
        print(json.dumps(metrics, indent=4, ensure_ascii=False))

        # Log persistente
        with open("results/evaluation_summary.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

        print(f"✨ Total Score: {metrics['accuracy']*100:.1f}% Accuracy")
        return results

    def _evaluate_single_question(self, question: str, expected_answer: str) -> EvaluationResult:
        """Evalúa una pregunta individual."""
        response = self.process_query(question)
        result = self.evaluator.evaluate(question, response, expected_answer)
        return result
