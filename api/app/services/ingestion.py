import re
from pathlib import Path

# from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, Filter, FilterSelector, VectorParams

from app.core.config import get_embeddings, get_qdrant_client, get_settings

# Text that marks a handbook section as internal / staff-only.
STAFF_MARKERS = ("staff only", "internal:", "must not reveal", "do not reveal")

# A heading is a markdown heading (## ...) or a numbered section (12. Title).
_HEADING_RE = re.compile(r"^(#{1,6}\s|\d+\.\s+\S)", re.MULTILINE)


def _is_staff(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in STAFF_MARKERS)


def strip_staff_sections(documents: list[Document]) -> list[Document]:
    """Remove staff-only sections from each page before chunking.

    A staff section starts at the heading that mentions a staff marker and runs
    until the next heading (or the end of the page). Dropping it here keeps
    internal facts out of the customer index entirely, so they can never be
    retrieved or leaked no matter how the chunker splits the text.
    """
    cleaned: list[Document] = []
    for doc in documents:
        text = doc.page_content
        low = text.lower()

        positions = [low.find(m) for m in STAFF_MARKERS if m in low]
        if not positions:
            cleaned.append(doc)
            continue

        start = min(positions)
        # Back up to the start of the heading line that opens the staff section.
        line_start = text.rfind("\n", 0, start) + 1
        # Find the next heading after the staff section (if any) to resume.
        next_match = _HEADING_RE.search(text, start + 1)
        tail = text[next_match.start():] if next_match else ""

        new_text = (text[:line_start] + tail).strip()
        if new_text:
            doc.page_content = new_text
            cleaned.append(doc)
        # If the whole page was staff-only, drop it entirely.
    return cleaned


def load(path: Path) -> list[Document]:
    return PyPDFLoader(path).load()


def ingest(path: str | None = None) -> int:
    settings = get_settings()

    # 1. Load
    file = Path(path or settings.data_path)
    documents = load(file)

    # 1b. Drop staff-only sections so they never enter the customer index.
    if not settings.index_staff_content:
        documents = strip_staff_sections(documents)

    # 2. Chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        raise ValueError(f"No text found in {file}")

    # Tag every chunk's audience so retrieval can filter staff-only content
    # as a backstop (relevant only when index_staff_content is enabled).
    for chunk in chunks:
        chunk.metadata["audience"] = "staff" if _is_staff(chunk.page_content) else "customer"

    # 3. Embed: check the model works before the old collection is deleted
    embed = get_embeddings(settings)

    actual_dim = len(embed.embed_query("dimension check"))
    if actual_dim != settings.embed_dim:
        raise ValueError(
            f"RAG_EMBED_DIM is {settings.embed_dim}, "
            f"but {settings.embed_model} returns {actual_dim} dimensions"
        )

    # 4. Store
    client = get_qdrant_client(settings)

    try:
        if client.collection_exists(settings.collection_name):
            # In local (path) mode, delete_collection does not drop the on-disk
            # segments, so recreating reloads stale vectors. Clear every point
            # explicitly instead to guarantee a fresh index each run.
            client.delete(
                collection_name=settings.collection_name,
                points_selector=FilterSelector(filter=Filter()),
            )
        else:
            client.create_collection(
                collection_name=settings.collection_name,
                vectors_config=VectorParams(
                    size=settings.embed_dim, distance=Distance.COSINE
                ),
            )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.collection_name,
            embedding=embed,
        )
        vector_store.add_documents(chunks)
    finally:
        client.close()

    return len(chunks)

if __name__ == "__main__":
    print(ingest(), "chunks indexed")
