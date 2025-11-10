from src.interface.base_datastore import BaseDatastore
from src.interface.base_retriever import BaseRetriever
import cohere
from cohere.errors import TooManyRequestsError
from dotenv import load_dotenv
import os
import time

class Retriever(BaseRetriever):
    def __init__(self, datastore: BaseDatastore):
        self.datastore = datastore

    def search(self, query: str, top_k: int = 10) -> list[str]:
        search_results = self.datastore.search(query, top_k=top_k * 3)
        reranked_results = self._rerank(query, search_results, top_k=top_k)
        return reranked_results

    def _rerank(
        self, query: str, search_results: list[str], top_k: int = 10
    ) -> list[str]:
        load_dotenv()
        co_api_key=os.getenv("CO_API_KEY")
        co = cohere.ClientV2(co_api_key)
        time.sleep(6)
        for attempt in range(5):  # hasta 5 reintentos
            try:
                time.sleep(10)  # evita pasarte del límite
                response = co.rerank(
                    model="rerank-multilingual-v3.5",
                    query=query,
                    documents=search_results,
                    top_n=top_k,
                )
                break
            except TooManyRequestsError:
                wait = 10 * (attempt + 1)
                print(f"⚠️ Límite de Cohere alcanzado. Reintentando en {wait}s...")
                time.sleep(wait)
        else:
            raise RuntimeError("Demasiados intentos fallidos con Cohere")

        result_indices = [result.index for result in response.results]
        print(f"✅ Reranked Indices: {result_indices}")
        return [search_results[i] for i in result_indices]