from tavily import TavilyClient
from dotenv import load_dotenv
import os

load_dotenv()

client = TavilyClient(
    api_key = os.getenv("TAVILY_API_KEY")
)

def tavily_search(query:str):
    """Search the web using Tavily"""
    response=client.search(query=query,max_results=5)
    result=[]
    for i, r in enumerate(response.get("results", [])):
        title= r.get("title","No title")
        url=r.get("url")
        snippet=r.get("content", "").strip()
        # keeps only the first 300 characters to avoid wall-of-text
        if len(snippet)>300:
            snippet=snippet[:300].rsplit(" ", 1)[0]+"..."
        
        result.append(
            f"[{i+1} Title:{title}]\nURL:{url}\n{snippet}"
        )
    
    return "\n\n".join(result)