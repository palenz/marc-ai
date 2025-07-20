import streamlit as st
from openai import OpenAI
import pickle
import os
import numpy as np
from pathlib import Path

# Show title and description.
st.title("📄 Marc AI")
st.write(
    "Hazme cualquier pregunta sobre la normativa técnica."
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
# Load pre-computed embeddings from file
# -----------------------------------------------------------------------------

EMBEDDINGS_FILE = "embeddings.pkl"
EMBED_MODEL = "text-embedding-3-small"

@st.cache_data
def load_embeddings_data():
    """Load pre-computed embeddings and chunks from pickle file."""
    if not Path(EMBEDDINGS_FILE).exists():
        st.error(
            f"❌ Embeddings file '{EMBEDDINGS_FILE}' not found. "
            "Please run `python process_pdfs_llamaparse.py` first to create the embeddings."
        )
        st.stop()
    
    try:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            data = pickle.load(f)
        
        st.success(f"✅ Loaded {data['total_chunks']} chunks from {len(set(chunk['source_file'] for chunk in data['chunks']))} PDFs")
        return data
        
    except Exception as e:
        st.error(f"❌ Failed to load embeddings: {e}")
        st.stop()

# Load the embeddings data
embeddings_data = load_embeddings_data()
chunks = embeddings_data['chunks']
embeddings = embeddings_data['embeddings']

def format_chunk_for_display(chunk):
    """Format a chunk with proper source attribution including document title, exact filename, and public link."""
    # Use document title if available, otherwise fall back to filename
    title = chunk.get('document_title', chunk['source_file'])
    filename = chunk['source_file']
    
    # Create public link to Google Cloud Storage
    public_link = f"https://storage.googleapis.com/marcai/{filename}"
    
    source_info = f"[FUENTE: {title} (archivo: [{filename}]({public_link}))"
    if chunk.get('page_number', 1) > 1:
        source_info += f" – pág {chunk['page_number']}"
    source_info += "]"
    
    return f"{source_info}\n{chunk['text']}"

def search_similar_chunks(query: str, top_k: int = 5):
    """Find the most similar chunks to the query."""
    # Create embedding for the query
    query_embed = client.embeddings.create(
        model=EMBED_MODEL, 
        input=[query]
    ).data[0].embedding
    
    query_vec = np.array(query_embed, dtype="float32")
    query_vec /= np.linalg.norm(query_vec) + 1e-10
    
    # Calculate similarities
    similarities = np.dot(embeddings, query_vec)
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Return top chunks with their similarity scores
    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        score = similarities[idx]
        results.append({
            'chunk': chunk,
            'similarity': float(score),
            'formatted_text': format_chunk_for_display(chunk)
        })
    
    return results

# Fixed system prompt
SYSTEM_PROMPT = (
    "Eres un asistente virtual que responde preguntas sobre Normativa Técnica.\n"
    "TODAS tus respuestas se deben basar en los documentos a los que tienes acceso.\n"
    "Si no encuentras la respuesta, debes responder \"No he encontrado una respuesta en la normativa técnica.\"\n"
    "Con cada respuesta, SIEMPRE debes referenciar la fuente. Indica al usuario:\n"
    "- El TÍTULO completo del documento\n"
    "- El nombre EXACTO del archivo PDF\n"
    "- La página específica cuando sea posible\n"
    "- El capítulo, artículo o sección relevante si está disponible\n"
    "Formato de ejemplo: 'Según el Real Decreto 1027/2007 por el que se aprueba el Reglamento de Instalaciones Térmicas en los Edificios (archivo: RD_1027_2007_RITE.pdf), página 87, las instalaciones térmicas...'\n"
    "Si el contexto incluye el título del documento entre corchetes [FUENTE: ...], úsalo en tu respuesta incluyendo siempre el nombre del archivo.\n"
    "NOTA: Los documentos originales están disponibles como enlaces clickeables en las fuentes mostradas."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input for the next user question
user_input = st.chat_input("Pregúntame sobre la normativa técnica...")

if user_input:
    # Append user's message to history and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Search for relevant chunks
    with st.spinner("Buscando información relevante..."):
        search_results = search_similar_chunks(user_input, top_k=5)
    
    # Build context from top results
    context_parts = [result['formatted_text'] for result in search_results]
    context_text = "\n\n---\n\n".join(context_parts)

    # Build the message list for the OpenAI request
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Contexto de la normativa técnica:\n\n{context_text}"},
    ] + st.session_state.messages

    # Generate assistant response
    with st.spinner("Generando respuesta..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
            )
            assistant_reply = response.choices[0].message.content
        except Exception as e:
            assistant_reply = f"❌ Error al generar respuesta: {e}"

    # Display assistant response and store it
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
        
        # Show clickable source links after the response
        st.markdown("---")
        st.markdown("**📚 Fuentes consultadas:**")
        for i, result in enumerate(search_results, 1):
            chunk = result['chunk']
            title = chunk.get('document_title', chunk['source_file'])
            filename = chunk['source_file']
            public_link = f"https://storage.googleapis.com/marcai/{filename}"
            page = chunk.get('page_number', 1)
            
            st.markdown(f"{i}. [{title}]({public_link}) - Página {page}")

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

    # Show debug information in an expander
    with st.expander("🔍 Información de búsqueda (debug)"):
        st.write(f"**Documentos consultados:**")
        for i, result in enumerate(search_results, 1):
            chunk = result['chunk']
            similarity = result['similarity']
            title = chunk.get('document_title', 'Sin título')
            element_category = chunk.get('element_category', 'Unknown')
            element_type = chunk.get('element_type', 'Unknown')
            filename = chunk['source_file']
            public_link = f"https://storage.googleapis.com/marcai/{filename}"
            
            st.write(f"{i}. **{title}** (pág {chunk['page_number']}) - Similitud: {similarity:.3f}")
            st.write(f"   📄 Archivo: [{filename}]({public_link})")
            st.write(f"   🏷️ Tipo: {element_category} ({element_type})")
            if st.checkbox(f"Ver texto completo {i}", key=f"show_chunk_{i}"):
                st.text_area(f"Contenido {i}:", chunk['text'], height=200, key=f"chunk_text_{i}")

# Sidebar with statistics
with st.sidebar:
    st.header("📊 Estadísticas")
    
    total_pdfs = len(set(chunk['source_file'] for chunk in chunks))
    total_chunks = len(chunks)
    
    st.metric("PDFs procesados", total_pdfs)
    st.metric("Chunks totales", total_chunks)
    st.metric("Dimensiones embedding", embeddings.shape[1])
    
    st.header("📚 Documentos disponibles")
    pdf_files = sorted(set(chunk['source_file'] for chunk in chunks))
    for pdf_file in pdf_files:
        pdf_chunks = [c for c in chunks if c['source_file'] == pdf_file]
        st.write(f"• **{pdf_file}** ({len(pdf_chunks)} chunks)")
    
    st.header("🔧 Configuración")
    st.write(f"Modelo de embedding: `{EMBED_MODEL}`")
    st.write(f"Archivo de embeddings: `{EMBEDDINGS_FILE}`")
    
    if st.button("🔄 Recargar embeddings"):
        st.cache_data.clear()
        st.rerun()
