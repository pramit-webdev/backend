from fastapi import FastAPI, UploadFile, File
from app.ingest import process_file
from app.graph import build_graph
from app.vectorstore_hf import VectorStoreHF
import os

app = FastAPI()
graph = build_graph()

vector_store = VectorStoreHF(hf_token=os.getenv("HF_TOKEN"))

@app.post("/process/")
async def process_docs(marketing: UploadFile = File(...),
                       sales: UploadFile = File(...),
                       product: UploadFile = File(...)):

    files = {"Marketing": marketing, "Sales": sales, "Product": product}
    docs_chunks = {}
    for dept, f in files.items():
        content = await f.read()
        chunks = process_file(content, f.filename)
        docs_chunks[dept] = chunks
        vector_store.add_texts(chunks)

    state = {"summaries": {}}
    for dept, chunks in docs_chunks.items():
        combined_text = "\n".join(chunks)
        out = graph.invoke({"dept": dept, "text": combined_text})
        state["summaries"][dept] = out["summary"]

    comparison = graph.invoke({"summaries": state["summaries"]})
    state.update(comparison)

    insights = graph.invoke({"comparison": state["comparison"]})
    state.update(insights)

    explanation = graph.invoke({"insights": state["insights"]})
    state.update(explanation)

    return state

@app.post("/query/")
async def query_docs(question: str):
    results = vector_store.search(question, k=3)
    prompt = f"Answer the following question based on these documents:\n\n{results}\n\nQuestion: {question}"
    from app.groq_llm import call_llm
    answer = call_llm(prompt)
    return {"answer": answer, "sources": results}
