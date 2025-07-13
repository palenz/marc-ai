import streamlit as st
from openai import OpenAI
import glob
import os
import numpy as np
import tiktoken

# Show title and description.
st.title("📄 Marc AI")
st.write(
    "Ask any question about the built-in reference documents below, and GPT will answer using ONLY that context."
)

# Retrieve the OpenAI API key from Streamlit secrets or environment variable.
openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error(
        "OpenAI API key not configured. Please set `OPENAI_API_KEY` in your environment "
        "or add it to `.streamlit/secrets.toml`."
    )
    st.stop()

# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# -----------------------------------------------------------------------------
# Build a lightweight vector-store (embeddings + metadata) the first time the
# app runs, then keep it cached in st.session_state for subsequent queries.
# -----------------------------------------------------------------------------

CHUNK_TOKENS = 200  # aprox. 200 tokens ≈ 1 página en muchos PDF convertidos
EMBED_MODEL = "text-embedding-3-small"

if ("embeddings" not in st.session_state) or (st.session_state.get("chunk_tokens") != CHUNK_TOKENS):
    st.info("Indexando normativa técnica (puede tardar unos minutos)…")

    encoder = tiktoken.get_encoding("cl100k_base")
    chunks, metadata, embeddings = [], [], []

    # 1. Split every .txt into ~CHUNK_TOKENS token chunks
    for path in glob.glob("data/text/*.txt"):
        with open(path, "r", encoding="utf-8") as f:
            full_text = f.read()

        # Extraer título: primera línea no vacía
        first_line = next((ln.strip() for ln in full_text.splitlines() if ln.strip()), os.path.basename(path))

        tokens = encoder.encode(full_text)
        for i in range(0, len(tokens), CHUNK_TOKENS):
            token_slice = tokens[i : i + CHUNK_TOKENS]
            chunk_text = encoder.decode(token_slice)

            # ---- Añadimos línea de referencia real ----
            page_num = i // CHUNK_TOKENS + 1  # aproximación de página
            ref_line = f"[FUENTE: {first_line} – pág {page_num}]"
            chunk_text = ref_line + "\n" + chunk_text

            chunks.append(chunk_text)
            metadata.append({"source": first_line, "page": page_num})

    # 2. Embed in manageable batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunks[i : i + batch_size])
        embeddings.extend([d.embedding for d in resp.data])

    # 3. L2-normalise for cosine similarity and cache
    embeddings = np.array(embeddings, dtype="float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10

    st.session_state["chunks"] = chunks
    st.session_state["metadata"] = metadata
    st.session_state["embeddings"] = embeddings
    st.session_state["encoder"] = encoder
    st.session_state["chunk_tokens"] = CHUNK_TOKENS

    st.success("Indexación terminada")

# Fixed system prompt (not editable by the user).
SYSTEM_PROMPT = (
    "Eres un asistente virtual que responde preguntas sobre Normativa Técnica.\n"
    "TODAS tus respuestas se deben basar en los documentos a los que tienes acceso.\n"
    "Si no encuentras la respuesta, debes responder \"No he encontrado una respuesta en la normativa técnica.\"\n"
    "Con cada respuesta, SIEMPRE debes referenciar la fuente. Indicale al usuario la referencia de cada hecho que mencionas (TITULO del documento, seccion, capitulo, articulo etc). Incluye la página también. Por ejemplo: Fuente: Real Decreto 1027/2007, de 20 de julio, por el que se aprueba el Reglamento de Instalaciones Térmicas en los Edificios. Apéndice 1, página 87. Instalación térmica."
)

# Initialise chat history.
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display existing chat history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input for the next user question.
user_input = st.chat_input("Ask something about the documents")

if user_input:
    # Append user's message to history and display it.
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ---------------- Retrieve the most relevant chunks ---------------------
    query_embed = client.embeddings.create(model=EMBED_MODEL, input=[user_input]).data[0].embedding
    query_vec = np.array(query_embed, dtype="float32")
    query_vec /= np.linalg.norm(query_vec) + 1e-10

    similarities = np.dot(st.session_state["embeddings"], query_vec)
    top_idx = similarities.argsort()[-5:][::-1]  # Top-5 most similar chunks

    context_parts = [st.session_state["chunks"][idx] for idx in top_idx]
    context_text = "\n\n---\n\n".join(context_parts)

    # Build the message list for the OpenAI request.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Contexto:\n\n{context_text}"},
    ] + st.session_state.messages

    # Generate assistant response.
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )
    assistant_reply = response.choices[0].message.content

    # Display assistant response and store it.
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
