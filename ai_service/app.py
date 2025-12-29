import requests
import os
import chromadb
import sqlite3
import subprocess
import re
from flask import Flask, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL")

app = Flask(__name__)

analyzer = SentimentIntensityAnalyzer()
model = SentenceTransformer("all-MiniLM-L6-v2")

chroma = chromadb.Client(
    chromadb.config.Settings(
        persist_directory="../data/chroma"
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
    conn = sqlite3.connect("../data/messages.db")
    rows = conn.execute(
        "SELECT text FROM messages WHERE chat_id=?",
        (chat_id,)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]

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
    embeddings = model.encode(messages)

    collection.add(
        documents=messages,
        embeddings=embeddings.tolist(),
        ids=[f"{chat_id}-{i}" for i in range(len(messages))]
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
    app.run(port=8000)
