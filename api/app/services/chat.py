from langchain.chat_models import init_chat_model
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel

from core.config import get_settings, get_embeddings, get_qdrant_client

NO_ANSWER = "I'm sorry, I don't have an answer to that question."
class Source(BaseModel):
    source: str
    page: int  | None = None
    score: float | None = None

class Answer(BaseModel):
    answer: str
    sources: list[Source] | None = None

class RAGService():
    def __init__(self):
        self.settings = get_settings()
        self.client = get_qdrant_client(settings=self.settings)

        if not self.client.collection_exists(collection_name=self.settings.collection_name):
            raise RuntimeWarning(f"Collection {self.settings.collection_name} does not exist in Qdrant. Please create it first.")

        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=self.settings.collection_name,
            embedding=get_embeddings(settings=self.settings)
        )
        self.llm = init_chat_model(
            model=self.settings.chat_model,
            model_provider=self.settings.chat_provider
        )

    def ask(self, query: str) -> Answer:
        question = query.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        # Use similarity_search_with_score to get scores for the Source model
        results = self.store.similarity_search_with_score(query=question, k=self.settings.top_k)

        if not results:
            return Answer(answer=NO_ANSWER, sources=[])

        context = "\n\n".join(doc.page_content for doc, score in results)

        messages = [
            {"role": "system", "content": self.settings.chat_system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]

        response = self.llm.invoke(input=messages)
        # Handle different response types from different LLM providers
        answer_text = getattr(response, 'content', str(response))

        return Answer(
            answer=answer_text,
            sources=[
                Source(
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page"),
                    score=round(score, 4)
                )
                for doc, score in results
            ]
        )

    def close(self) -> None:
        pass
