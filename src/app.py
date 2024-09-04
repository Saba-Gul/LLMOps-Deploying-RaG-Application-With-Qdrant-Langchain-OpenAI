import sys
sys.path.append("C:\\Users\\Saba Gul\\Downloads\\LLMOps-main\\LLMOps-main")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.main import retriever

app = FastAPI()

class QueryRequest(BaseModel):
    question: str



@app.post("/ask/")
async def ask_question(query: QueryRequest):
    # Validate the input
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Call the main logic
    answer = retriever(query.question)
    
    return {"question": query.question, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
    #  run "uvicorn app:app --reload" in the command line