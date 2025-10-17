from pydantic import BaseModel

class ChatRequest(BaseModel):
    """
    Represents the request body for the /chat endpoint.
    """
    user_prompt: str
    session_id: str | None = None # Optional: for tracking conversation history

class ChatResponse(BaseModel):
    """
    Represents the response body for the /chat endpoint.
    """
    final_answer: str
    fact_check_passed: bool
    explanation: str | None = None
