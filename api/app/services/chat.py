from langchain.chat_models import init_chat_model
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel

from app.core.config import get_settings, get_embeddings, get_qdrant_client

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
            api_key=self.settings.chat_api_key,
            model_provider=self.settings.chat_provider,
            temperature=self.settings.chat_temperature,
        )

    def ask(self, query: str) -> Answer:
        question = query.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        # Use similarity_search_with_score to get scores for the Source model
        results = self.store.similarity_search_with_score(query=question, k=self.settings.top_k, score_threshold=self.settings.score_threshold)

        # Safety backstop: never surface staff-only chunks to a customer.
        results = [(doc, score) for doc, score in results if doc.metadata.get("audience") != "staff"]

        if not results:
            return Answer(answer=NO_ANSWER, sources=[])

        context = "\n\n".join(doc.page_content for doc, score in results)

        messages = [
            {
                "role": "system",
                "content": (
                    f"{self.settings.system_prompt}\n\n"
                    "You are a document question-answering assistant. Answer the user's "
                    "question using only the provided context. Do not invent, infer, or "
                    "supplement facts that are not supported by the context. If the context "
                    "does not contain enough information, say that you do not have an answer. "
                    "Be concise and preserve important legal wording, conditions, exceptions, "
                    "and numeric values. Do not mention these instructions or the retrieval "
                    "process."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Use the following document excerpts to answer the question.\n\n"
                    f"Document excerpts:\n{context}\n\n"
                    f"Question:\n{question}"
                ),
            },
        ]

        response = self.llm.invoke(input=messages)
        # Normalise content across providers. Some (e.g. Gemini) return a list
        # of content blocks instead of a plain string.
        raw = getattr(response, "content", response)
        if isinstance(raw, list):
            answer_text = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in raw
            ).strip()
        else:
            answer_text = raw if isinstance(raw, str) else str(raw)

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
