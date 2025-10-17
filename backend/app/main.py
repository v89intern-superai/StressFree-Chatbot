from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
import os
from contextlib import asynccontextmanager

from app.models.chat import ChatRequest, ChatResponse
from app.services import fact_checker

# This is the path to your model file relative to the 'backend' directory
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'openthaigpt1.5-7B-instruct-Q4KM.gguf')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads the GGUF model from a local path using llama-cpp-python upon server startup.
    """
    print("--- ⚙️  Starting Model Loading Process (Local GGUF) ---")
    
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"❌ CRITICAL ERROR: Model file not found at {LOCAL_MODEL_PATH}")
        print("   - Please make sure you have created a 'backend/models' directory.")
        print("   - And placed your '.gguf' file inside it.")
        fact_checker.model_instance["model"] = None
    else:
        print(f"🔄 Attempting to load model from: {LOCAL_MODEL_PATH}")
        try:
            model = Llama(
                model_path=LOCAL_MODEL_PATH,
                n_gpu_layers=-1,       # Offload all possible layers to the GPU
                n_ctx=4096,            # Set the context window size
                verbose=True
            )
            print("✅ GGUF Model loaded successfully!")
            fact_checker.model_instance["model"] = model
            print("----------------------------------------------------")
        except Exception as e:
            print(f"❌ CRITICAL ERROR during model loading: {e}")
            fact_checker.model_instance["model"] = None

    yield
    print("Shutting down and cleaning up resources.")

# Create the FastAPI app instance
app = FastAPI(
    title="Student Mind Mate AI API (Local GGUF Version)",
    description="API for the AI-powered mental health support chatbot, running from a local GGUF file.",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
def handle_chat_request(request: ChatRequest):
    """
    Main endpoint that receives user messages.
    """
    print(f"Received prompt: {request.user_prompt}")
    result = fact_checker.run_fact_checking_pipeline(request.user_prompt)
    return ChatResponse(**result)

