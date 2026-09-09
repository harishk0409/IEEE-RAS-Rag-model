"""
app.py
Streamlit RAG chat assistant for IEEE RAS VIT Chennai.

Local run:
    streamlit run app.py

Deployment (Streamlit Community Cloud):
    1. Push this whole folder (including chroma_db/ after running ingest.py) to GitHub
    2. On streamlit.io/cloud, create a new app pointing at this repo, main file app.py
    3. In "Secrets", add: GROQ_API_KEY = "your_key_here"
    4. Deploy -> you get a public URL
"""

import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# ---------- CONFIG ----------
DB_DIR = "chroma_db"
COLLECTION_NAME = "ieee_ras_docs"
LLM_MODEL = "llama-3.1-8b-instant"   # fast + free on Groq
TOP_K = 4

st.set_page_config(page_title="IEEE RAS VIT Chennai Assistant", page_icon="🤖")
st.title("🤖 IEEE RAS VIT Chennai Assistant")
st.caption("Ask me about IEEE RAS VIT Chennai — events, workshops, hackathons, and how to get involved.")


# ---------- LOAD RESOURCES (cached) ----------
@st.cache_resource
def load_collection():
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


@st.cache_resource
def load_groq_client():
    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to Streamlit secrets or environment variables.")
        st.stop()
    return Groq(api_key=api_key)


collection = load_collection()
groq_client = load_groq_client()


# ---------- RAG LOGIC ----------
def retrieve_context(query, k=TOP_K):
    results = collection.query(query_texts=[query], n_results=k)
    chunks = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]
    return chunks, sources


def generate_answer(query, chunks):
    context_text = "\n\n---\n\n".join(chunks)
    system_prompt = (
        "You are the official IEEE RAS VIT Chennai assistant. "
        "Answer the user's question using ONLY the context provided below. "
        "Be friendly, concise, and specific (mention event names, dates, numbers when relevant). "
        "If the answer isn't in the context, say you don't have that information yet, "
        "and suggest checking the official IEEE RAS VIT Chennai page: "
        "https://edu.ieee.org/in-rasvitcc/\n\n"
        f"CONTEXT:\n{context_text}"
    )

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return response.choices[0].message.content


# ---------- CHAT UI ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm the IEEE RAS VIT Chennai assistant. Ask me about our events, workshops, or how to get involved!"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask about IEEE RAS VIT Chennai...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chunks, sources = retrieve_context(user_query)
            answer = generate_answer(user_query, chunks)
            st.markdown(answer)

            with st.expander("📄 Sources used"):
                for s in set(sources):
                    st.write(f"- {s}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
