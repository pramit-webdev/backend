from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from app.ingest import process_file
from app.graph import build_graph
from app.vectorstore_openai import VectorStoreOpenAI
from app.openai_llm import call_llm
import os

# Initialize FastAPI app
app = FastAPI(title="DocuSync API", version="1.0")

# Initialize Graph and Vector Store safely
try:
    graph = build_graph()
except Exception as e:
    print(f"[Graph Init] Failed: {e}")
    graph = None

try:
    vector_store = VectorStoreOpenAI()
except Exception as e:
    print(f"[VectorStore Init] Failed: {e}")
    vector_store = None


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok", "message": "API is running"}


@app.post("/process/")
async def process_docs(files: List[UploadFile] = File(...)):
    """
    Accept multiple files and process them.
    Safe: handles failures in LangGraph and always returns a response.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    docs_chunks = {}
    for f in files:
        content = await f.read()
        chunks = process_file(content, f.filename)
        docs_chunks[f.filename] = chunks
        if vector_store:
            try:
                vector_store.add_texts(chunks)
            except Exception as e:
                print(f"[VectorStore] Failed to add texts for {f.filename}: {e}")

    state = {"summaries": {}}

    # Generate summaries per document safely
    for dept, chunks in docs_chunks.items():
        combined_text = "\n".join(chunks)
        if graph:
            try:
                out = graph.invoke({"dept": dept, "text": combined_text})
                summary = out.get("summary", "No summary generated")
            except Exception as e:
                print(f"[Graph Invoke] Failed for {dept}: {e}")
                summary = "Error generating summary"
        else:
            summary = "Graph not initialized"
        state["summaries"][dept] = summary

    # Generate comparison
    if graph:
        try:
            comparison = graph.invoke({"summaries": state.get("summaries", {})})
            state.update(comparison)
        except Exception as e:
            print(f"[Graph Comparison] Failed: {e}")
            state["comparison"] = {}

    # Generate insights
    if graph:
        try:
            insights = graph.invoke({"comparison": state.get("comparison", {})})
            state.update(insights)
        except Exception as e:
            print(f"[Graph Insights] Failed: {e}")
            state["insights"] = {}

    # Generate explanation
    if graph:
        try:
            explanation = graph.invoke({"insights": state.get("insights", {})})
            state.update(explanation)
        except Exception as e:
            print(f"[Graph Explanation] Failed: {e}")
            state["explanation"] = {}

    return state


@app.post("/query/")
async def query_docs(question: str):
    """
    Search vector store and get answer from LLM
    """
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    results = []
    if vector_store:
        try:
            results = vector_store.search(question, k=3)
        except Exception as e:
            print(f"[VectorStore Search] Failed: {e}")

    prompt = f"Answer the following question based on these documents:\n\n{results}\n\nQuestion: {question}"
    answer = "LLM not available"
    try:
        answer = call_llm(prompt)
    except Exception as e:
        print(f"[LLM] Failed to get answer: {e}")

    return {"answer": answer, "sources": results}


# Render requires dynamic PORT binding
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
