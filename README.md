# TechMart AI Shopping Assistant & Concierge (Clara)

An intelligent, full-stack e-commerce AI concierge and shopping assistant powered by **Groq LLMs**, **ChromaDB Semantic Vector Search**, live **Google Sheets** synchronization, and **Gmail API** order processing.

---

## ✨ Features

- **Clara AI Persona**: Friendly, accurate, and conversational sales concierge following strict source-of-truth guidelines.
- **In-Memory Chat History**: Per-user session isolation with zero disk/database persistence and 30-minute idle TTL expiration.
- **Query-Rewriting Retrieval**: Reformulates conversational follow-ups (e.g. *"tell me more about the Samsung one"*) into standalone queries before database lookup.
- **Live Google Sheets Synchronization**: Real-time read and write access for product catalogs, customer orders, and refund requests.
- **ChromaDB Semantic Policy RAG**: Natural language retrieval over return policies, warranties, and shipping rules.
- **Automated Email Confirmations**: Sends itemized HTML order confirmations and secret tracking passkeys via Gmail API.

---

## 🚀 Quick Start (Local)

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/arhamakhan798-code/ai-agent.git
cd ai-agent
pip install -r requirements.txt
```

### 2. Configure Credentials
Copy `.env.example` to `.env` or place your Google Cloud Service Account JSON and Gmail token in the project root:
- `techmart-database-*.json`
- `gmail_token.json`
- Set `GROQ_API_KEY` and `GEMINI_API_KEY`

### 3. Run the Backend
```bash
python app.py
```
Or double-click `start_agent.bat` on Windows.

### 4. Open the Web Widget
Open `widget.html` in your browser to start chatting with Clara!

---

## 🌐 Cloud Deployment (Render / Railway / Heroku)

1. Connect your GitHub repository `arhamakhan798-code/ai-agent`.
2. Set the build command:
   ```bash
   pip install -r requirements.txt
   ```
3. Set the start command:
   ```bash
   gunicorn app:app --bind 0.0.0.0:$PORT
   ```
4. Set the Environment Variables in your cloud dashboard:
   - `GROQ_API_KEY`
   - `GEMINI_API_KEY`
   - `GOOGLE_SERVICE_ACCOUNT_JSON` (paste the content of your service account JSON file)
   - `GMAIL_TOKEN_JSON` (paste the content of your `gmail_token.json` file)

---

## 📄 License
MIT License
