import requests
import os
import chromadb
import subprocess
import re
from supabase import create_client, Client
from flask import Flask, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)

analyzer = SentimentIntensityAnalyzer()
model = SentenceTransformer("all-MiniLM-L6-v2")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_data")
os.makedirs(CHROMA_DIR, exist_ok=True)

chroma = chromadb.Client(
    chromadb.config.Settings(
        persist_directory=CHROMA_DIR
    )
)
collection = chroma.get_or_create_collection("chat")

client = InferenceClient(
    model="Qwen/Qwen3-1.7B",
    api_key=HF_API_TOKEN
)

def clean_output(text: str) -> str:
    # Remove <think>...</think> blocks
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def get_messages(chat_id):
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
    messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant.\n"
            "Use the provided context in the form of telegram text messages to answer the question.\n"
            "Do NOT reveal your chain-of-thought, reasoning, or internal analysis.\n"
            "Provide ONLY the final answer in clear, concise language."
        )
    },
    {
        "role": "user",
        "content": f"""
        Context:
        {context}

        Question:
        {question}

        Give only the final answer.
        """
    }
    ]

    response = client.chat.completions.create(
        messages=messages,
        max_tokens=300,
        temperature=0.3,
    )

    raw_answer = response.choices[0].message.content
    return clean_output(raw_answer)

@app.route("/sentiment", methods=["POST"])
def sentiment():
    text = request.json["text"]
    score = analyzer.polarity_scores(text)["compound"]

    label = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
    return jsonify({"sentiment": label})

@app.route("/ask", methods=["POST"])
def ask():
    chat_id = request.json["chat_id"]
    question = request.json["question"]

    messages = get_messages(chat_id)
    if not messages:
        return jsonify({"answer": "No chat history found for this chat."})

    embeddings = model.encode(messages)

    # only add new messages to ChromaDB
    existing_ids = set(collection.get()["ids"])  # all existing IDs
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

if __name__ == "__main__":
    # To run locally
    # source venv/bin/activate LOCAL=1 python3 app.py
    if os.environ.get("LOCAL") == "1":
        port = int(os.environ.get("PORT", 8000))
        app.run(host="0.0.0.0", port=port)
