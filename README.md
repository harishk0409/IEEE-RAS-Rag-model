# IEEE RAS VIT Chennai — RAG Assistant

A retrieval-augmented chatbot that answers questions about IEEE RAS VIT Chennai using its own event history and about-page content.

## How it works
1. `data/*.txt` — raw knowledge base (about page + event recaps)
2. `ingest.py` — chunks the text, embeds it locally (sentence-transformers), stores vectors in ChromaDB (`chroma_db/`)
3. `app.py` — Streamlit chat UI: embeds your question, retrieves the most relevant chunks, sends them + your question to a free Groq LLM (Llama 3.1), and shows the answer with sources

## Run locally
```bash
pip install -r requirements.txt
python ingest.py          # builds the vector database (run once, or after editing data/)
streamlit run app.py      # starts the chat app
```
You'll need a free Groq API key from https://console.groq.com — set it as an environment variable:
```bash
export GROQ_API_KEY="your_key_here"
```

## Deploy (free) — Streamlit Community Cloud
1. Push this whole folder to a GitHub repo — **including the `chroma_db/` folder** (run `ingest.py` locally first so it exists)
2. Go to https://share.streamlit.io → "New app" → pick your repo → main file `app.py`
3. In the app's **Settings → Secrets**, add:
   ```
   GROQ_API_KEY = "your_key_here"
   ```
4. Deploy. You'll get a public URL like `https://your-app.streamlit.app` — that's your working deployed application link.

## Adding more data
Drop more `.txt` or `.md` files into `data/` (e.g. more event recaps, FAQ, how-to-join info), then re-run `python ingest.py` and redeploy/push.
