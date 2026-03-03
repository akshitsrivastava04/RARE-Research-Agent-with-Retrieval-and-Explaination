from pydantic import BaseModel
from typing import Optional, List

# ---------------- Chat ----------------

class ChatRequest(BaseModel):
    """
    Request model for sending a question to the AI council.
    Optionally includes a list of document IDs for context retrieval.
    """
    question: str
    document_ids: Optional[List[str]] = None

class ChatResponse(BaseModel):
    """
    Response model for chat endpoint.
    Contains the final consensus answer from all agents.
    """
    answer: str

# ---------------- Document Upload ----------------

class UploadDocumentRequest(BaseModel):
    """
    Request model for uploading a new document to the vector database.
    The document text is extracted and stored in Qdrant as embeddings.
    """
    text: str
    doc_id: str

class UploadDocumentResponse(BaseModel):
    """
    Response model for successful document upload.
    Returns the document ID and status.
    """
    status: str
    doc_id: str
