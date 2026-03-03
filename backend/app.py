import os
import uuid
import datetime
import asyncio
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from bson import ObjectId
import numpy as np
import faiss

# --- FastAPI & Auth ---
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# --- MongoDB ---
import motor.motor_asyncio

# --- Security (Passwords & JWT) ---
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, UTC

# --- AI & RAG Imports (Unchanged) ---
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from services.groq_api import GroqAPI
from services.gemini_summarizer import summarize_context
from services.search_service import SearchService

# =======================================================================
# --- NEW: SECURITY & CONFIGURATION ---
# =======================================================================

# --- Load from .env file ---
from dotenv import load_dotenv
load_dotenv()

# --- JWT Settings ---
MONGO_DETAILS = os.getenv("MONGO_DETAILS", "mongodb://localhost:27017")
SECRET_KEY = os.getenv("SECRET_KEY", "a_very_secret_key_change_this_asap")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 24 hours

# --- Password Hashing ---
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# --- OAuth2 Scheme ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login", auto_error=False)

# =======================================================================
# --- NEW: MONGODB SETUP ---
# =======================================================================

USE_MOCK_DB = os.getenv("USE_MOCK_DB", "false").lower() == "true"

if USE_MOCK_DB:
    print("⚠️ USING MOCK MONGODB (Memory Only) ⚠️")
    try:
        from mongomock_motor import AsyncMongoMockClient
        client = AsyncMongoMockClient()
    except ImportError:
        print("Mock DB requested but mongomock_motor not installed. Falling back to real DB.")
        client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
else:
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)

db = client.riya_ai_db 
user_collection = db.get_collection("users")
session_collection = db.get_collection("sessions")

# --- Helper for MongoDB ObjectId ---
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, v, *args, **kwargs):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema, *args, **kwargs):
        field_schema.update(type="string")

# ---------------- FastAPI App ----------------
app = FastAPI(title="RiyaAI Backend (Groq + Council Edition)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- AI Models & Utils ---
try:
    embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2", device="cpu")
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150, length_function=len
)
groq_clients: Dict[str, GroqAPI] = {}
search_service = SearchService(max_results=3)

def get_groq_client(model_id: str) -> GroqAPI:
    if model_id not in groq_clients:
        print(f"[Groq Client] Initializing new client for: {model_id}")
        try:
            groq_clients[model_id] = GroqAPI(model_id=model_id)
        except (ValueError, EnvironmentError) as e:
            print(f"[Groq Client Error] {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize model: {e}")
    return groq_clients[model_id]

# =======================================================================
# --- AUTH HELPERS ---
# =======================================================================

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
    return pwd_context.hash(password_bytes)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# =======================================================================
# --- PYDANTIC MODELS (Moved Up) ---
# =======================================================================

class UserSchema(BaseModel):
    username: str
    password: str

class UserInDB(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    username: str
    hashed_password: str

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class Message(BaseModel):
    role: str
    text: str

class ChatSessionSchema(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: PyObjectId = Field(...)
    title: str = Field(...)
    text_chunks: List[str] = Field(default_factory=list)
    messages: List[Message] = Field(default_factory=list)
    createdAt: datetime = Field(default_factory=lambda: datetime.now(UTC))

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class UploadRequest(BaseModel):
    text: str
    session_id: Optional[str] = None

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    model_id: str
    upload_text: Optional[str] = None
    web_search: bool = False

class UploadResponse(BaseModel):
    session_id: str
    status: str

class ModelUsage(BaseModel):
    total_tokens_used: int

class UsageData(BaseModel):
    date: str
    models: Dict[str, ModelUsage]

class ChatResponse(BaseModel):
    answer: str
    tokens_from_this_call: int
    usage_data: UsageData
    session_id: str

# =======================================================================
# --- USER DEPENDENCY (Uses Models Defined Above) ---
# =======================================================================

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    """
    Validates the token. 
    DEBUG MODE: If no token is provided, returns a temporary 'Guest' user to allow development without login.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not token:
        # print("[DEBUG] No token provided. Using 'Guest' user.")
        # Try to find or create a guest user
        guest_user = await user_collection.find_one({"username": "guest"})
        if not guest_user:
            guest_id = ObjectId()
            guest_user = {
                "_id": guest_id,
                "username": "guest",
                "hashed_password": get_password_hash("guest"),
            }
            try:
                await user_collection.insert_one(guest_user)
            except Exception:
                pass # concurrency
        return UserInDB(**guest_user)

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = await user_collection.find_one({"username": token_data.username})
    if user is None:
        raise credentials_exception
    
    return UserInDB(**user)
# =======================================================================
# --- AUTHENTICATION ENDPOINTS ---
# =======================================================================

@app.post("/register", response_model=UserSchema)
async def register_user(user: UserSchema):
    existing_user = await user_collection.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    
    user_document = {
        "username": user.username,
        "hashed_password": hashed_password
    }
    
    try:
        new_user = await user_collection.insert_one(user_document)
    except Exception:
         raise HTTPException(status_code=500, detail="Database error")
    
    return user 

@app.post("/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await user_collection.find_one({"username": form_data.username})
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me")
async def get_me(current_user: UserInDB = Depends(get_current_user)):
    return {"username": current_user.username}

# =======================================================================
# --- SESSION DATA ENDPOINTS ---
# =======================================================================

@app.get("/sessions", response_model=List[ChatSessionSchema])
async def get_all_sessions(current_user: UserInDB = Depends(get_current_user)):
    """
    Fetches all chat sessions for the logged-in user.
    """
    sessions = []
    cursor = session_collection.find({"user_id": current_user.id}).sort("createdAt", -1)
    async for session in cursor:
        sessions.append(ChatSessionSchema(**session))
    return sessions

@app.get("/sessions/{session_id}", response_model=ChatSessionSchema)
async def get_session(session_id: str, current_user: UserInDB = Depends(get_current_user)):
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session ID format")

    session = await session_collection.find_one({"_id": oid, "user_id": current_user.id})
    if session:
        return ChatSessionSchema(**session)
    raise HTTPException(status_code=404, detail="Session not found or not owned by user")

@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, current_user: UserInDB = Depends(get_current_user)):
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session ID format")

    result = await session_collection.delete_one({"_id": oid, "user_id": current_user.id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    return

# =======================================================================
# --- UPLOAD & CHAT ENDPOINTS ---
# =======================================================================

@app.post("/upload", response_model=UploadResponse)
async def upload_document(req: UploadRequest, current_user: UserInDB = Depends(get_current_user)):
    user_id = current_user.id
    new_chunks = text_splitter.split_text(req.text)
    if not new_chunks:
        raise HTTPException(status_code=400, detail="No text chunks found")

    try:
        if req.session_id:
            session_oid = ObjectId(req.session_id)
            result = await session_collection.update_one(
                {"_id": session_oid, "user_id": user_id},
                {"$push": {"text_chunks": {"$each": new_chunks}}}
            )
            if result.matched_count == 0:
                raise HTTPException(status_code=404, detail="Session not found")
            session_id = req.session_id
        else:
            title = new_chunks[0][:40] + "..."
            new_session = ChatSessionSchema(user_id=user_id, title=title, text_chunks=new_chunks)
            result = await session_collection.insert_one(new_session.model_dump(by_alias=True, exclude=["id"]))
            session_id = str(result.inserted_id)
        return UploadResponse(session_id=session_id, status="success")
    except Exception as e:
        print(f"Error upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")

async def _process_and_save_chunks(text: str, user_id: PyObjectId, session_id: Optional[str] = None) -> Tuple[str, ObjectId, List[str]]:
    new_chunks = text_splitter.split_text(text)
    if not new_chunks:
        raise HTTPException(status_code=400, detail="No text chunks found")

    if session_id:
        session_oid = ObjectId(session_id)
        result = await session_collection.update_one(
            {"_id": session_oid, "user_id": user_id},
            {"$push": {"text_chunks": {"$each": new_chunks}}}
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Session not found")
        return session_id, session_oid, new_chunks
    else:
        title = new_chunks[0][:40] + "..."
        new_session = ChatSessionSchema(user_id=user_id, title=title, text_chunks=new_chunks)
        result = await session_collection.insert_one(new_session.model_dump(by_alias=True, exclude=["id"]))
        session_oid = result.inserted_id
        return str(session_oid), session_oid, new_chunks

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, current_user: UserInDB = Depends(get_current_user)):
    user_id = current_user.id
    session_id = req.session_id
    session_oid = None
    text_chunks = []
    
    try:
        if req.upload_text:
            session_id, session_oid, text_chunks = await _process_and_save_chunks(req.upload_text, user_id, session_id)
        elif session_id:
            session_oid = ObjectId(session_id)
            session = await session_collection.find_one({"_id": session_oid, "user_id": user_id})
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            text_chunks = session.get("text_chunks", [])
        else:
            title = req.question[:40] + "..."
            new_session = ChatSessionSchema(user_id=user_id, title=title)
            result = await session_collection.insert_one(new_session.model_dump(by_alias=True, exclude=["id"]))
            session_oid = result.inserted_id
            session_id = str(session_oid)

        user_message = Message(role="user", text=req.question)
        await session_collection.update_one({"_id": session_oid}, {"$push": {"messages": user_message.model_dump()}})
        
        summary = ""
        if text_chunks:
            try:
                chunk_embeddings = embedding_model.encode(text_chunks, normalize_embeddings=True).astype(np.float32)
                index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
                index.add(chunk_embeddings)
                q_emb = embedding_model.encode([req.question], normalize_embeddings=True).astype(np.float32)
                D, I = index.search(q_emb, 3)
                retrieved = [text_chunks[i] for i in I[0] if i != -1 and i < len(text_chunks)]
                summary = await summarize_context("\n\n".join(retrieved))
            except Exception as e:
                print(f"RAG Error: {e}")

        search_context = ""
        if req.web_search:
            print(f"[Web Search] Fetching context for: {req.question}")
            search_context = await search_service.get_search_context(req.question)
            if search_context:
                search_context = f"\n--- WEB SEARCH RESULTS ---\n{search_context}\n"

        question_with_context = (f"Ctx: {summary}\n{search_context}\nQ: {req.question}" if (summary or search_context) else req.question)
        client = get_groq_client(req.model_id)
        answer = await client.query(question_with_context)

        bot_message = Message(role="bot", text=answer)
        await session_collection.update_one({"_id": session_oid}, {"$push": {"messages": bot_message.model_dump()}})

        tokens = 0
        if client.last_response_metadata:
             tokens = client.last_response_metadata.get('token_usage', {}).get('total_tokens', 0)
        daily = await client.get_current_model_daily_total()
        
        return ChatResponse(
            answer=answer, 
            tokens_from_this_call=tokens,
            usage_data={"date": datetime.now(UTC).date().isoformat(), "models": {req.model_id: {"total_tokens_used": daily}}},
            session_id=session_id
        )
    except Exception as e:
        print(f"Chat Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# =======================================================================
# --- NEW: COUNCIL CHAT ENDPOINT ---
# =======================================================================

from agents.agent import Agent

class CouncilChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    # We will use hardcoded agents for now as per agent.py example, 
    # but strictly speaking these could be configured.

class CouncilResponse(BaseModel):
    individual_responses: List[Dict[str, str]]
    final_answer: str
    tokens_used: int
    usage_data: UsageData
    session_id: str

@app.post("/council_chat", response_model=CouncilResponse)
async def council_chat(
    req: CouncilChatRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Executes the Council flow:
    1. Initializes 3 agents (Cookie, Captcha, Plag).
    2. Runs the council (Split responses -> Judge).
    3. Saves the final result to the chat history.
    """
    user_id = current_user.id
    session_id = req.session_id
    session_oid = None
    
    try:
        # --- 1. Find or Create Session ---
        if session_id:
            session_oid = ObjectId(session_id)
            session = await session_collection.find_one({"_id": session_oid, "user_id": user_id})
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            # Create new session
            new_session = ChatSessionSchema(
                user_id=user_id,
                title=f"Council: {req.question[:30]}..."
            )
            result = await session_collection.insert_one(new_session.model_dump(by_alias=True, exclude=["id"]))
            session_oid = result.inserted_id
            session_id = str(session_oid)

        # --- 2. Save User Message ---
        user_message = Message(role="user", text=req.question)
        await session_collection.update_one(
            {"_id": session_oid},
            {"$push": {"messages": user_message.model_dump()}}
        )
        
        # --- 3. Initialize Agents ---
        print("Initializing Council Agents...")
        agent1 = Agent("Cookie", "Summarizer", model_id="llama-3.1-8b-instant")
        agent2 = Agent("Captcha", "QA", model_id="moonshotai/kimi-k2-instruct")
        agent3 = Agent("Plag", "Critic", model_id="qwen/qwen3-32b")
        
        # Dedicated Judge as requested
        judge = Agent("Judge", "Consensus Specialist", model_id="openai/gpt-oss-20b")
        
        # --- 4. Run Council ---
        # The Judge overseeing all three agents
        result = await judge.run_council(req.question, [agent1, agent2, agent3])
        
        final_answer = result["answer"]
        individual_responses = result["individual_responses"]
        tokens_used = result["tokens_used"]
        
        # --- 5. Save Bot Message (The Final Verdict) ---
        # We only save the final answer to history for now, but we could save the whole council log if needed
        bot_message = Message(role="bot", text=final_answer)
        await session_collection.update_one(
            {"_id": session_oid},
            {"$push": {"messages": bot_message.model_dump()}}
        )
        
        # --- 6. Usage Data ---
        usage_data = {
            "date": datetime.now(UTC).date().isoformat(),
            "models": {
                "council_total": {"total_tokens_used": tokens_used}
            }
        }
        
        return CouncilResponse(
            individual_responses=individual_responses,
            final_answer=final_answer,
            tokens_used=tokens_used,
            usage_data=usage_data,
            session_id=session_id
        )

    except Exception as e:
        print(f"Error during council chat: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

if __name__ == "__main__":
    import uvicorn
    # Make sure to bind to 0.0.0.0 so external devices can access it if needed
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
