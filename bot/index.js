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
  console.log("RECEIVED:", msg.text);

  if (!msg.text) return;

  const chatId = msg.chat.id;
  const text = msg.text;

  // Handle /sentiment
  if (text.startsWith("/sentiment")) {
    const cleanedText = text.replace("/sentiment", "").trim();
    await saveMessage(chatId, cleanedText);

    try {
      const res = await axios.post(`${process.env.AI_SERVICE_URL}/sentiment`, { text: cleanedText });
      bot.sendMessage(chatId, `Sentiment: ${res.data.sentiment}`);
    } catch (err) {
      console.error(err);
      bot.sendMessage(chatId, "Error processing request.");
    }
  }

  // Handle /ask
  else if (text.startsWith("/ask")) {
    const cleanedText = text.replace("/ask", "").trim();
    await saveMessage(chatId, cleanedText);

    try {
      const res = await axios.post(`${process.env.AI_SERVICE_URL}/ask`, { chat_id: chatId, question: cleanedText });
      bot.sendMessage(chatId, res.data.answer);
    } catch (err) {
      console.error(err);
      bot.sendMessage(chatId, "Error processing request.");
    }
  }

  // Save other messages
  else {
    await saveMessage(chatId, text);
  }
});

console.log("🤖 Bot running");
