# LAPI FastAPI

from fastapi import FastAPI # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from pydantic import BaseModel # type: ignore
from rag_engine import get_query_engine

app = FastAPI(title="Rentaload Chatbot API", version="1.0")

# Configuration de CORS pour permitir le connexion hostgator
origins = [
    "http://rentaload.cl"
    "https://www.rentaload.cl",
    "http://localhost:5173",       # Para pruebas locales si lo necesitas
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # O los dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

query_engine = get_query_engine()

class ChatRequest(BaseModel):
    message: str
    
@app.get("/")
def read_root():
    return {"message": "API de Rentaload funcionando correctamente"}
@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    response = query_engine.query(request.message)
    return {
        "reply": str(response)
    }