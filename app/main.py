from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from pydantic import BaseModel

from app.ingest import process_file
from app.vectorstore_openai import VectorStoreOpenAI
from app.openai_llm import call_llm
from app.agents import summarizer_agent, comparator_agent, insight_agent, explainer_agent

app = FastAPI(title="DocuSync API", version="1.0")

# Initialize vector store
vector_store = VectorStoreOpenAI()


# -------------------------------
# Pydantic model for /query/ JSON
# -------------------------------
class QueryRequest(BaseModel):
    question: str


# -------------------------------
# Health check
# -------------------------------
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok", "message": "API is running"}


# -------------------------------
# Process multiple files
# -------------------------------
@app.post("/process/")
async def process_docs(files: List[UploadFile] = File(...)):
    """
    Accept multiple files and process them sequentially through the agents:
    summarizer -> comparator -> insights -> explainer
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

    # Step 1: Summarize each document
    for dept, chunks in docs_chunks.items():
        combined_text = "\n".join(chunks)
        try:
            summary = summarizer_agent({"dept": dept, "text": combined_text})["summary"]
        except Exception as e:
            summary = f"Error generating summary: {e}"
        state["summaries"][dept] = summary

    # Step 2: Compare summaries
    try:
        comparison = comparator_agent({"summaries": state["summaries"]})["comparison"]
    except Exception as e:
        comparison = f"Error generating comparison: {e}"
    state["comparison"] = comparison

    # Step 3: Generate insights
    try:
        insights = insight_agent({"comparison": comparison})["insights"]
    except Exception as e:
        insights = f"Error generating insights: {e}"
    state["insights"] = insights

    # Step 4: Explain insights
    try:
        explanation = explainer_agent({"insights": insights})["explanation"]
    except Exception as e:
        explanation = f"Error generating explanation: {e}"
    state["explanation"] = explanation

    return state


# -------------------------------
# Query endpoint (JSON)
# -------------------------------
@app.post("/query/")
async def query_docs(request: QueryRequest):
    """
    Search vector store and get answer from LLM
    """
    question = request.question
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    results = vector_store.search(question, k=3)
    prompt = f"Answer the following question based on these documents:\n\n{results}\n\nQuestion: {question}"
    answer = call_llm(prompt)
    return {"answer": answer, "sources": results}
