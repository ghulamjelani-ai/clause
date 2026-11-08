from app.services.ingestion import ingest
from app.services.chat import RAGService


# print(f"Ingested {ingest()} chunks")


def main() -> None:
    rag_service = RAGService()
    try:
        query=input("You: ").strip()
        result = rag_service.ask(query=query)
        print(f"Bot: {result.answer}")
    
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rag_service.close()

if __name__ == "__main__":
    main()
