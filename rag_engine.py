import os
from dotenv import load_dotenv # type: ignore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings # type: ignore
from llama_index.llms.google_genai import GoogleGenAI # type: ignore
from llama_index.core.node_parser import SentenceSplitter # type: ignore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # type: ignore

# 1. Carga las variables del archivo .env
load_dotenv()

# EXTRAER TU LLAVE PERSONALIZADA Y ASIGNARLA DONDE LA API LA ESPERA
api_key = os.getenv("RENTALOAD_GOOGLE_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GEMINI_API_KEY"] = api_key
else:
    raise ValueError("No se encontró la variable RENTALOAD_GOOGLE_API_KEY en el archivo .env")

# Prompt del sistema para guiar el comportamiento del chatbot
SYSTEM_PROMPT = """
Eres el asistente virtual oficial de RentaLoad, empresa especializada en pruebas isocinéticas, pruebas de capacidad de generadores y arriendo de bancos de carga en Cerrillos, Santiago, Chile.

REGLAS DE RESPUESTA:
1. Sé extremadamente breve y directo. Responde en un máximo de 2 a 3 líneas o viñetas cortas. Evita rodeos, saludos largos o explicaciones excesivas. Ve siempre al grano.
2. Responde de forma profesional, técnica, amable y comercial basándote estrictamente en los servicios de RentaLoad (bancos de carga hasta 3 MW, pruebas isocinéticas y pruebas de rendimiento para generadores; no hacemos climatización).
3. Si el usuario pregunta por cotizaciones, visitas técnicas, arriendos o contratación de servicios, DEBES incluir obligatoriamente los canales de contacto y, AL FINAL DE TODO, agregar de manera explícita la siguiente frase con los datos:

"Para coordinar tu servicio o solicitar una cotización formal, contáctanos directamente al correo lhenriquez@rentaload.cl, al teléfono +569 9826 4750"
"""

# 2. Configurar Gemini como LLM (usando el modelo flash correcto)
Settings.llm = GoogleGenAI(model="gemini-3-flash-preview", system_instruction=SYSTEM_PROMPT)

# 3. Configurar el modelo de embeddings local
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

DATA_DIR = "./data"
STORAGE_DIR = "./storage"

def get_query_engine():
    if not os.path.exists(STORAGE_DIR) or not os.listdir(STORAGE_DIR):
        # Leer documentos desde la carpeta ./data
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        
        # Control explícito de los chunks (fragmentos de texto)
        parser = SentenceSplitter(chunk_size=200, chunk_overlap=20)
        nodes = parser.get_nodes_from_documents(documents)        
        
        # Crear el índice vectorial a partir de los nodos (chunks)
        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=STORAGE_DIR)
    else:
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=STORAGE_DIR))
    
    return index.as_query_engine(response_mode="compact")