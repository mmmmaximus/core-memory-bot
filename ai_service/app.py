import os
import re
import requests
import chromadb

from flask import Flask, request, jsonify
from supabase import create_client, Client
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# -------------------------------------------------------------------
# ENV
# -------------------------------------------------------------------
load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# -------------------------------------------------------------------
# Flask app
# -------------------------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------------------------
# Lazy-loaded globals (IMPORTANT)
# -------------------------------------------------------------------
analyzer = None
model = None
chroma = None
collection = None
client = None
supabase: Client | None = None

# -------------------------------------------------------------------
# Lazy initialization
# -------------------------------------------------------------------
def init_services():
    global analyzer, model, chroma, collection, client, supabase

    if analyzer is None:
        analyzer = SentimentIntensityAnalyzer()

    if supabase is None:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    if chroma is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        chroma_dir = os.path.join(base_dir, "chroma_data")
        os.makedirs(chroma_dir, exist_ok=True)

        chroma = chromadb.Client(
            chromadb.config.Settings(
                persist_directory=chroma_dir
            )
        )
        collection = chroma.get_or_create_collection("chat")

    if client is None:
        client = InferenceClient(
            model=HF_MODEL,
            api_key=HF_API_TOKEN
        )

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def clean_output(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def get_messages(chat_id):
    init_services()

    response = (
        supabase
        .table("messages")
        .select("text")
        .eq("chat_id", str(chat_id))
        .order("created_at")
        .execute()
    )

    return [row["text"] for row in response.data]

def generate_answer(context, question):
    init_services()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant.\n"
                "Use the provided context to answer the question.\n"
                "Provide ONLY the final answer."
            )
        },
        {
            "role": "user",
            "content": f"""
Context:
{context}

Question:
{question}
"""
        }
    ]

    response = client.chat.completions.create(
        messages=messages,
        max_tokens=300,
        temperature=0.3,
    )

    return clean_output(response.choices[0].message.content)

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.route("/", methods=["GET"])
def health():
    return "OK", 200

@app.route("/sentiment", methods=["POST"])
def sentiment():
    init_services()

    text = request.json["text"]
    score = analyzer.polarity_scores(text)["compound"]

    label = (
        "positive" if score > 0.05
        else "negative" if score < -0.05
        else "neutral"
    )

    return jsonify({"sentiment": label})

@app.route("/ask", methods=["POST"])
def ask():
    init_services()

    chat_id = request.json["chat_id"]
    question = request.json["question"]

    messages = get_messages(chat_id)
    if not messages:
        return jsonify({"answer": "No chat history found for this chat."})

    embeddings = model.encode(messages)

    existing_ids = set(collection.get()["ids"])
    new_docs, new_embeddings, new_ids = [], [], []

    for i, msg in enumerate(messages):
        msg_id = f"{chat_id}-{i}"
        if msg_id not in existing_ids:
            new_docs.append(msg)
            new_embeddings.append(embeddings[i].tolist())
            new_ids.append(msg_id)

    if new_docs:
        collection.add(
            documents=new_docs,
            embeddings=new_embeddings,
            ids=new_ids
        )

    q_emb = model.encode([question])[0]
    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=5
    )

    context = "\n".join(results["documents"][0])
    answer = generate_answer(context, question)

    return jsonify({"answer": answer})

# -------------------------------------------------------------------
# Local run (ignored by Gunicorn on Render)
# -------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
