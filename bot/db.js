const Database = require("better-sqlite3");
const db = new Database("../data/messages.db");

db.prepare(`
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id TEXT,
  user TEXT,
  text TEXT,
  timestamp INTEGER
)
`).run();

module.exports = db;
