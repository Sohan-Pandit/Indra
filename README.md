# Indra

**Distill Extreme Events Research Abstracts**

Indra is a web application that extracts structured climate impact data from scientific research abstracts using LLMs. Paste any abstract about an extreme weather event and get back a structured JSON record with hazard type, location, impact domain, uncertainty analysis, and more.

---

## Features

- Structured extraction of climate impact metadata from research abstracts
- Supports multiple LLM providers — bring your own API key
- Uncertainty analysis and secondary impact detection
- Clean, fast React frontend

## Supported API Keys

| Provider | Key Format |
|---|---|
| Google Gemini | `AIza...` |
| Anthropic Claude | `sk-ant-...` |
| OpenAI | `sk-...` |
| Grok (xAI) | `xai-...` |
| Groq | `gsk_...` |

---

## Run Locally

### Frontend

**Prerequisites:** Node.js

```bash
npm install
npm run dev
```

### Backend

**Prerequisites:** Python 3.10+

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

The frontend runs on `http://localhost:3000` and calls the backend at `http://localhost:8001`.

---

## Project Structure

```
indra/
├── backend/
│   ├── main.py               # FastAPI app
│   ├── requirements.txt
│   └── src/
│       ├── annotation/       # LLM labeling pipeline
│       ├── extraction/       # NER models
│       ├── llm/              # Unified LLM wrapper
│       ├── schema/           # Impact schema & validation
│       ├── uncertainty/      # Hedge detection
│       └── visualization/    # Knowledge graph
├── components/               # React components
├── services/                 # API service layer
├── App.tsx
└── index.tsx
```

---

## Built by Sohan Pandit
