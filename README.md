# Indra

Extract structured climate impact data from research abstracts.

Paste an abstract → get back hazard type, location, impact domain, uncertainty analysis, and a full JSON record. Bring your own API key (Gemini, OpenAI, Anthropic, Grok, Groq).

---

## Run locally

**Frontend**
```bash
npm install
npm run dev
```

**Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --port 8001 --reload
```

Frontend → `localhost:3000`, Backend → `localhost:8001`

---

## Project layout

```
├── backend/
│   ├── main.py
│   └── src/
│       ├── annotation/
│       ├── llm/
│       ├── schema/
│       └── uncertainty/
├── components/
├── services/
└── App.tsx
```
