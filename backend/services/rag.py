from services.document_store import DocumentStore

class RAGSystem:
    """
    Retrieval-Augmented Generation (RAG) system.
    Uses the DocumentStore (Qdrant) to retrieve relevant context
    for a given question, then formats it for agents.
    """

    def __init__(self, doc_store: DocumentStore, top_k: int = 5):
        """
        :param doc_store: Instance of DocumentStore
        :param top_k: Number of relevant documents to retrieve per query
        """
        self.doc_store = doc_store
        self.top_k = top_k

    def get_context(self, question: str) -> str:
        """
        Retrieves top_k relevant documents from the vector database
        and returns them concatenated as context for the agents.

        :param question: The user's query
        :return: A single string containing the context
        """
        # NEW:
        retrieved_texts = self.doc_store.search(question, limit=self.top_k)
        if not retrieved_texts:
            return ""
        # Merge retrieved documents into a single context block
        context = "\n\n".join(retrieved_texts)
        return context

    def get_context_with_metadata(self, question: str) -> list[dict]:
        """
        Optional: Retrieve not only text but also metadata from documents.
        Useful if you want to include source information in prompts.

        :param question: The user's query
        :return: List of dicts with 'text' and 'metadata'
        """
        retrieved_texts = self.doc_store.search(question, limit=self.top_k)
        results = []
        for text in retrieved_texts:
            results.append({
                "text": text,
                "metadata": {"length": len(text)}
            })
        return results
