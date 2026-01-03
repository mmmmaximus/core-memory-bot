import TelegramBot from "node-telegram-bot-api";
import axios from "axios";
import { createClient } from "@supabase/supabase-js";
import dotenv from "dotenv";

dotenv.config();

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_KEY
);

// Telegram polling bot
const bot = new TelegramBot(process.env.BOT_TOKEN, { polling: true });

// Save messages to Supabase
const saveMessage = async (chat_id, text) => {
  await supabase.from("messages").insert({
    chat_id: String(chat_id),
    text: text,
  });
}

bot.on("message", async (msg) => {
  if (!msg.text) return;

  const chatId = msg.chat.id;
  let text = msg.text;
  if (text.startsWith("/ask")) {
    text = text.replace("/ask", "").trim();

    try {
      const res = await axios.post(`${process.env.AI_SERVICE_URL}/ask`, { chat_id: chatId, question: text });
      bot.sendMessage(chatId, res.data.answer);
    } catch (err) {
      bot.sendMessage(chatId, "Error processing request.");
    }
  }

  // Ingest EVERY message into the Vector DB via the Python service
  try {
    await axios.post(`${process.env.AI_SERVICE_URL}/ingest`, { chat_id: chatId, text: text });
  } catch (err) {
    console.error("Vector Ingestion Error:", err.message);
  }
});

console.log("🤖 Bot running");
