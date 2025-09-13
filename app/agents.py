from app.groq_llm import call_llm

def summarizer_agent(state):
    dept, text = state["dept"], state["text"]
    prompt = f"Summarize the following {dept} document in concise bullet points:\n\n{text}"
    return {"summary": call_llm(prompt)}

def comparator_agent(state):
    summaries = state["summaries"]
    combined = "\n\n".join([f"{k}: {v}" for k, v in summaries.items()])
    prompt = f"Compare the following departmental summaries and find overlaps, gaps, contradictions:\n\n{combined}"
    return {"comparison": call_llm(prompt)}

def insight_agent(state):
    comparison = state["comparison"]
    prompt = f"From this comparison, extract risks, opportunities, and alignment issues:\n\n{comparison}"
    return {"insights": call_llm(prompt)}

def explainer_agent(state):
    insights = state["insights"]
    prompt = f"Explain these insights in simple, actionable recommendations for leadership:\n\n{insights}"
    return {"explanation": call_llm(prompt)}
