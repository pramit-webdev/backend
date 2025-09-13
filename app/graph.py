from langgraph.graph import StateGraph, END
from app.agents import summarizer_agent, comparator_agent, insight_agent, explainer_agent

def build_graph():
    g = StateGraph(dict)

    g.add_node("summarizer", summarizer_agent)
    g.add_node("comparator", comparator_agent)
    g.add_node("insight", insight_agent)
    g.add_node("explainer", explainer_agent)

    g.set_entry_point("summarizer")
    g.add_edge("summarizer", "comparator")
    g.add_edge("comparator", "insight")
    g.add_edge("insight", "explainer")
    g.add_edge("explainer", END)

    return g.compile()
