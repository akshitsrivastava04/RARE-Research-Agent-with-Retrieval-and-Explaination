#!/usr/bin/env python3
import os
import sys
import yaml
import glob
import json
import time
import requests
import typer
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pypdf.errors import PdfStreamError, PdfReadError
from langchain import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from pypdf import PdfReader

app = typer.Typer(help="CrewAI CLI agent (RAG)")

load_dotenv()

def load_config(config_path: str = "config.yaml") -> dict:
    if not Path(config_path).exists():
        raise FileNotFoundError(f"{config_path} not found. Create it from the provided template.")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("hf_api_token", os.environ.get("HF_API_TOKEN", cfg.get("hf_api_token", "")))
    cfg.setdefault("lmstudio_api_key", os.environ.get("LMSTUDIO_API_KEY", cfg.get("lmstudio_api_key", "")))
    return cfg

def read_pdf_text(path: str) -> str:
    try:
        reader = PdfReader(path)
        pages = []
        for p in reader.pages:
            txt = p.extract_text() or ""
            pages.append(txt)
        return "\n".join(pages)
    except (PdfStreamError, PdfReadError, Exception) as e:
        print(f"[WARN] Skipping {path} due to error: {e}")
        return ""

def load_documents_from_dir(docs_dir: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    docs: List[Document] = []
    p = Path(docs_dir)
    if not p.exists():
        raise FileNotFoundError(f"Docs directory {docs_dir} not found.")
    files = list(p.rglob("*.*"))
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for fp in files:
        if fp.suffix.lower() in [".pdf"]:
            text = read_pdf_text(str(fp))
        elif fp.suffix.lower() in [".txt", ".md", ".text"]:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        else:
            continue
        if not text or text.strip() == "":
            continue
        splits = text_splitter.split_text(text)
        for i, part in enumerate(splits):
            metadata = {"source": str(fp), "chunk": i}
            docs.append(Document(page_content=part, metadata=metadata))
    return docs

@app.command()
def index(config: Optional[str] = typer.Option("config.yaml", help="Path to config.yaml")):
    cfg = load_config(config)
    docs_dir = cfg.get("docs_dir", "./data/docs")
    persist_directory = cfg.get("persist_directory", "./chroma_db")
    embedding_model = cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size = cfg.get("chunk_size", 800)
    chunk_overlap = cfg.get("chunk_overlap", 150)
    collection_name = cfg.get("collection_name", "crewai_collection")

    typer.echo(f"[INDEX] Loading docs from: {docs_dir}")
    documents = load_documents_from_dir(docs_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    typer.echo(f"[INDEX] Loaded {len(documents)} document chunks.")

    typer.echo(f"[INDEX] Creating embeddings ({embedding_model}) and saving to Chroma at {persist_directory} ...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    try:
        vectordb.persist()
    except Exception:
        pass

    typer.echo("[INDEX] Done. Vector store persisted.")

def _get_vectorstore(cfg: dict) -> Chroma:
    persist_directory = cfg.get("persist_directory", "./chroma_db")
    embedding_model = cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    collection_name = cfg.get("collection_name", "crewai_collection")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name=collection_name)
    return vectordb

def _make_hfhub_llm(cfg: dict):
    api_token = cfg.get("hf_api_token")
    model = cfg.get("hf_model")
    if not api_token:
        raise RuntimeError("HF API token required for huggingface_hub backend. Set HF_API_TOKEN in .env or config.yaml")

    llm = HuggingFaceEndpoint(
        repo_id=model,
        huggingfacehub_api_token=api_token,
        temperature=0.7,
        max_new_tokens=512,
    )
    return llm

def _make_local_transformers_llm(cfg: dict):
    model_name = cfg.get("local_model_name", "meta-llama/Llama-3.1-8B")
    max_new_tokens = int(cfg.get("local_max_new_tokens", 512))
    typer.echo("[LLM] Loading local transformers model (this may be slow).")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto" if torch.cuda.is_available() else None, torch_dtype=dtype)
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1, max_new_tokens=max_new_tokens)
    llm = HuggingFacePipeline(pipeline=gen)
    return llm

from langchain.llms.base import LLM
class LMStudioHTTP(LLM):
    def __init__(self, url: str, api_key: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 512):
        self.url = url
        self.api_key = api_key
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    @property
    def _llm_type(self) -> str:
        return "lmstudio_http"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"prompt": prompt, "max_tokens": self.max_tokens, "temperature": self.temperature}
        resp = requests.post(self.url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        j = resp.json()
        if isinstance(j, dict):
            if "generated_text" in j:
                return j["generated_text"]
            if "choices" in j and isinstance(j["choices"], list) and len(j["choices"]) > 0:
                c = j["choices"][0]
                return c.get("text") or c.get("message", {}).get("content") or json.dumps(j)
            if "data" in j and isinstance(j["data"], list) and "generated_text" in j["data"][0]:
                return j["data"][0]["generated_text"]
        return resp.text

    def _identifying_params(self):
        return {"url": self.url}

def _get_llm(cfg: dict):
    backend = cfg.get("backend", "huggingface_endpoint")
    if backend == "huggingface_endpoint":
        return _make_hfhub_llm(cfg)
    elif backend == "local_transformers":
        return _make_local_transformers_llm(cfg)
    elif backend == "lmstudio":
        url = cfg.get("lmstudio_url")
        if not url:
            raise RuntimeError("lmstudio_url missing in config for lmstudio backend.")
        api_key = cfg.get("lmstudio_api_key")
        return LMStudioHTTP(url=url, api_key=api_key, temperature=cfg.get("temperature", 0.0), max_tokens=cfg.get("max_tokens", 512))
    else:
        raise RuntimeError(f"Unknown backend {backend}")

def format_sources(source_docs: List[Document]) -> str:
    out = []
    seen = set()
    for d in source_docs:
        src = d.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            snippet = (d.page_content[:400] + "...") if len(d.page_content) > 400 else d.page_content
            out.append(f"Source: {src}\n---\n{snippet}\n")
    return "\n\n".join(out)

@app.command()
def query(question: str = typer.Argument(..., help="A single question to ask the RAG agent"),
          config: Optional[str] = typer.Option("config.yaml", help="Path to config.yaml")):
    cfg = load_config(config)
    vectordb = _get_vectorstore(cfg)
    retriever = vectordb.as_retriever(search_kwargs={"k": int(cfg.get("top_k", 4))})
    llm = _get_llm(cfg)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    typer.echo("[QUERY] Sending to RAG chain...")
    out = qa({"query": question})
    answer = out.get("result") or out.get("answer") or out
    typer.echo("\n===== Answer =====\n")
    typer.echo(answer)
    srcs = out.get("source_documents")
    if srcs:
        typer.echo("\n===== Sources (snippets) =====\n")
        typer.echo(format_sources(srcs))

@app.command()
def chat(config: Optional[str] = typer.Option("config.yaml", help="Path to config.yaml")):
    cfg = load_config(config)
    vectordb = _get_vectorstore(cfg)
    retriever = vectordb.as_retriever(search_kwargs={"k": int(cfg.get("top_k", 4))})
    llm = _get_llm(cfg)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conv = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    typer.echo("[CHAT] Starting interactive chat. Type 'exit' or Ctrl+C to quit.")
    try:
        while True:
            q = typer.prompt("You")
            if q.strip().lower() in {"exit", "quit"}:
                typer.echo("Goodbye.")
                break
            res = conv({"question": q})
            answer = res.get("answer") or res.get("result") or str(res)
            typer.echo("\nAssistant:\n")
            typer.echo(answer)
            srcs = res.get("source_documents")
            if srcs:
                typer.echo("\nSources:\n")
                typer.echo(format_sources(srcs))
                typer.echo("\n---\n")
    except KeyboardInterrupt:
        typer.echo("\nInterrupted. Goodbye.")
        sys.exit(0)

if __name__ == "__main__":
    app()
