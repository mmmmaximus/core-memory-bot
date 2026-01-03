import os
import re
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# Safety settings for deployment
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

# ENV
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

app = Flask(__name__)

# Lazy-loaded globals
model: SentenceTransformer | None = None
hf_client: InferenceClient | None = None
supabase: Client | None = None

def init_services():
    global model, hf_client, supabase
    if supabase is None:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    if model is None:
        # Load embedding model once
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    if hf_client is None:
        hf_client = InferenceClient(model=HF_MODEL, api_key=HF_API_TOKEN)

def clean_output(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def generate_answer(context: str, question: str) -> str:
    init_services()
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the context to answer. Provide ONLY the final answer."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
    ]
    response = hf_client.chat.completions.create(
        messages=messages,
        max_tokens=300,
        temperature=0.3,
    )
    return clean_output(response.choices[0].message.content)

@app.route("/", methods=["GET"])
def health():
    return "OK", 200

@app.route("/ingest", methods=["POST"])
def ingest():
    """Endpoint called by Node.js for every message to generate embeddings and save to Supabase."""
    init_services()

    chat_id = str(request.json.get("chat_id"))
    text = request.json.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Generate embedding using the Sentence Transformer
    embedding = model.encode(text).tolist()

    # Insert message and embedding into Supabase
    try:
        supabase.table("messages").insert({
            "chat_id": chat_id,
            "text": text,
            "embedding": embedding
        }).execute()
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Ingestion Database Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    init_services()
    chat_id = str(request.json["chat_id"])
    question = request.json["question"]

    # Query vector similarity using the Supabase RPC function 'match_messages'
    q_emb = model.encode(question).tolist()

    try:
        rpc_resp = supabase.rpc("match_messages", {
            "query_embedding": q_emb,
            "match_threshold": 0.5,
            "match_count": 5,
            "p_chat_id": chat_id
        }).execute()

        if not rpc_resp.data:
            return jsonify({"answer": "I don't have enough chat history to answer that yet."})

        # Build context and generate answer
        context = "\n".join([row["text"] for row in rpc_resp.data])
        answer = generate_answer(context, question)
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"Ask Route Error: {e}")
        return jsonify({"answer": "Error retrieving relevant history."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
