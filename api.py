from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio

from agent import ask_agent

app = FastAPI(title="Web Query Agent")

class QueryRequest(BaseModel):
    query: str  # Input prompt from the frontend

class QueryResponse(BaseModel):
    answer: str  # Output response to the frontend

@app.post("/ask", response_model=QueryResponse)
async def ask(req: QueryRequest):
    """
    Endpoint to handle a single prompt from the frontend and return the output.
    """
    try:
        answer = await ask_agent(req.query)  # Process the query using the agent
        return QueryResponse(answer=answer)  # Return the response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # Handle errors gracefully