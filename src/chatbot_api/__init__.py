import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    """Represents a request containing a prompt for the chatbot."""
    prompt: str

@app.get("/ping")
def ping():
    """ Endpoint to check if the API is running."""
    return {"status": "alive"}

@app.post("/generate")
async def generate_text(request: PromptRequest):
    """ Endpoint to generate text based on the provided prompt."""
    # Call GPT model
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Reponse only in markdown format."},
            {"role": "user", "content": request.prompt}
            ]
        )
    # response = client.responses.create(
    #     model="gpt-4.1-mini",
    #     input=request.prompt
    # )
    return {
        "prompt": request.prompt,
        "output": response.choices[0].message.content 
        if response.choices else "No response generated."
        }
