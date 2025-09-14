from fastapi import FastAPI, UploadFile, File, HTTPException
from app.ingest import process_file
from app.graph import build_graph
from app.vectorstore_openai import VectorStoreOpenAI
from app.openai_llm import call_llm
from typing import List

app = FastAPI(title="DocuSync API", version="1.0")

graph = build_graph()
vector_store = VectorStoreOpenAI()


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok", "message": "API is running"}


@app.post("/process/")
async def process_docs(files: List[UploadFile] = File(...)):
    """
    Accept multiple files and process them.
    Flexible: works with any number of uploaded files.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    docs_chunks = {}
    for f in files:
        content = await f.read()
        chunks = process_file(content, f.filename)
        docs_chunks[f.filename] = chunks
        vector_store.add_texts(chunks)

    state = {"summaries": {}}
    for dept, chunks in docs_chunks.items():
        combined_text = "\n".join(chunks)
        out = graph.invoke({"dept": dept, "text": combined_text})
        state["summaries"][dept] = out["summary"]

    # Multi-step summarization & insights pipeline
    comparison = graph.invoke({"summaries": state["summaries"]})
    state.update(comparison)

    insights = graph.invoke({"comparison": state["comparison"]})
    state.update(insights)

    explanation = graph.invoke({"insights": state["insights"]})
    state.update(explanation)

    return state


@app.post("/query/")
async def query_docs(question: str):
    """
    Search vector store and get answer from LLM
    """
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    results = vector_store.search(question, k=3)
    prompt = f"Answer the following question based on these documents:\n\n{results}\n\nQuestion: {question}"
    answer = call_llm(prompt)
    return {"answer": answer, "sources": results}
