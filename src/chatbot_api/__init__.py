import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import json


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

def get_weather(location: str) -> str:
    # Dummy implementation of a weather function
    return f"The weather in {location} is sunny with a high of 25°C."

def get_weather_function_chat():
    return {
        "type": "function",
        "function": { # This property is removed from responses API
            "name": "get_weather",
            "description": "Get the weather for a location. Call this whenever you need to know the weather, for example when a customer asks 'What's the weather like in this city'",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "name of the location"
                    }
                },
                "required": ["location"],
                "additionalProperties": False
            },
            "strict": True
        }
    }


@app.post("/generate")
async def generate_text(request: PromptRequest):
    """ Endpoint to generate text based on the provided prompt."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Reponse only in markdown format."},
    ]
    # Call GPT model
    messages.append({"role": "user", "content": request.prompt})
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=messages,
        tools=[get_weather_function_chat()],
        )
    print(response)
    
    if response.choices[0].finish_reason == "tool_calls":
        tool_call = response.choices[0].message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        location = arguments.get("location")
        
        weather = get_weather(location)
        new_message = {
            "role": "tool",
            "content": json.dumps({"location": location, "weather": weather}),
            "tool_call_id": tool_call.id
        }

        messages.append(response.choices[0].message)
        messages.append(new_message)

        response2 = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=messages,
            tools=[get_weather_function_chat()]
        )
        print("Model Response2 = ",response2.choices[0].message.content)
        print("Finish Reason = ",response2.choices[0].finish_reason)
        
    return {
        "prompt": request.prompt,
        "output": response2.choices[0].message.content 
        if response2.choices else "No response generated.",
        "messages": messages
        }
