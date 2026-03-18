import os
import base64
from dotenv import load_dotenv
import streamlit as st
from fpdf import FPDF

from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

# ------------------ UI HEADER ------------------

title = "ECE RGUKT,RKV Chat Assistant"
logo_path = "pics&pdf/images.jpg"

def get_base64_image(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

st.markdown(f"""
<style>
.title-container {{
    display:flex;
    align-items:center;
    margin-bottom:30px;
}}
.title {{
    font-size:35px;
    font-weight:bold;
    color:gold;
    margin-right:15px;
}}
.logo {{
    width:40px;
}}
</style>

<div class="title-container">
    <div class="title">{title}</div>
    <img src="data:image/png;base64,{get_base64_image(logo_path)}" class="logo">
</div>
""", unsafe_allow_html=True)

# ------------------ LOAD VECTOR DB ------------------

@st.cache_resource
def load_embeddings_and_db():
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db

db = load_embeddings_and_db()
retriever = db.as_retriever(search_kwargs={"k": 3})

# ------------------ LLM ------------------

@st.cache_resource
def load_llm():
    return ChatGroq(
        temperature=0.4,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )

model = load_llm()

system_prompt = (
    "You are a chatbot for ECE RGUKT RK Valley. "
    "Answer ONLY from context. "
    "If unknown, say you don't know. "
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# ------------------ SESSION ------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False

user_logo = "pics&pdf/user.jpeg"
bot_logo = "pics&pdf/bot.jpeg"

# ------------------ PDF DOWNLOAD FUNCTION ------------------

def generate_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Chat History", ln=True, align="C")
    pdf.ln(5)

    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"{role}:", ln=True)

        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, msg["content"])
        pdf.ln(3)

    return pdf.output(dest='S').encode('latin-1', 'ignore')

# ------------------ SIDEBAR CONTROLS ------------------

with st.sidebar:
    st.header("Controls")
    
    # Clear History Button
    if st.button("Clear History", key="clear_history"):
        st.session_state.messages = []
        st.session_state.processing = False
        st.rerun()
    
    # Download Button (only if there are messages)
    if st.session_state.messages:
        pdf_data = generate_pdf()
        st.download_button(
            label="Download Chat",
            data=pdf_data,
            file_name="chat_history.pdf",
            mime="application/pdf",
            key="download_chat"
        )
    else:
        st.write("No history yet")

# ------------------ DISPLAY MESSAGES ------------------

# Display all messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=user_logo if message["role"] == "user" else bot_logo):
        st.markdown(message["content"])

# ------------------ INPUT AND RESPONSE ------------------

# Get user input
user_input = st.chat_input("Ask something...")

if user_input and not st.session_state.processing:
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message immediately
    with st.chat_message("user", avatar=user_logo):
        st.markdown(user_input)
    
    # Set processing flag
    st.session_state.processing = True
    
    # Generate assistant response
    with st.chat_message("assistant", avatar=bot_logo):
        with st.spinner("Thinking..."):
            try:
                res = rag_chain.invoke({"input": user_input})
                answer = res["answer"]
            except Exception as e:
                answer = f"Error: {str(e)}"
        
        st.markdown(answer)
    
    # Add assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Reset processing flag
    st.session_state.processing = False
    
    # Rerun to ensure UI is updated
    st.rerun()