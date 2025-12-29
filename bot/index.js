require("dotenv").config();
const TelegramBot = require("node-telegram-bot-api");
const axios = require("axios");
const db = require("./db");

const bot = new TelegramBot(process.env.BOT_TOKEN, { polling: true });

bot.on("message", async (msg) => {
  console.log("RECEIVED:", msg.text);

  if (!msg.text) return;

  db.prepare(`
    INSERT INTO messages (chat_id, user, text, timestamp)
    VALUES (?, ?, ?, ?)
  `).run(
    msg.chat.id.toString(),
    msg.from?.username || "unknown",
    msg.text,
    msg.date
  );
});

// Sentiment command
bot.onText(/^\/sentiment(?:@\w+)?\s+([\s\S]+)/i, async (msg, match) => {
  const text = match[1].trim();

  try {
    const res = await axios.post(`${process.env.AI_SERVICE_URL}/sentiment`, {
      text
    });

    bot.sendMessage(msg.chat.id, `Sentiment: ${res.data.sentiment}`);
  } catch (err) {
    console.error(err);
    bot.sendMessage(msg.chat.id, "Error processing request.");
  }
});


// Ask command
bot.onText(/^\/ask(?:@\w+)?\s+([\s\S]+)/i, async (msg, match) => {
  const question = match[1].trim();
  console.log("ASK:", question);

  try {
    const res = await axios.post(`${process.env.AI_SERVICE_URL}/ask`, {
      chat_id: msg.chat.id.toString(),
      question
    });

    bot.sendMessage(msg.chat.id, res.data.answer);
  } catch (err) {
    console.error(err);
    bot.sendMessage(msg.chat.id, "Error processing request.");
  }
});

console.log("🤖 Bot running");
