# ECE RGUKT RKV Chatbot

A Retrieval-Augmented Generation (RAG) based chatbot designed specifically for the Electronics and Communication Engineering (ECE) department at RGUKT RK Valley. The chatbot answers queries based on the provided context from ECE-related documents.

---

## 📋 Features

- **🎯 Context-Aware Responses** — Answers questions strictly from the provided ECE document context
- **🤖 Powered by LLM** — Uses Groq's Llama 3.3-70b for generating accurate responses
- **🔍 Vector Search** — Employs FAISS for efficient similarity search
- **💬 Chat Interface** — Clean and intuitive chat UI with user/bot avatars
- **📥 Download History** — Export chat conversations as PDF
- **🔄 Session Management** — Persistent chat history during the session

---

## 🛠️ Technology Stack

| Component    | Technology               |
|--------------|--------------------------|
| Frontend     | Streamlit                |
| LLM          | Groq (Llama 3.3-70b)     |
| Embeddings   | Cohere (embed-english-v3.0) |
| Vector Store | FAISS                    |
| Framework    | LangChain                |
| PDF Generation | fpdf                   |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Cohere API Key
- Groq API Key

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/acharan-tech-200037/ece_rgukt_rkv_chatbot_using_rag.git
   cd ece_rgukt_rkv_chatbot_using_rag
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the root directory:

   ```env
   COHERE_API_KEY=your_cohere_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Prepare the vector database**

   Ensure you have your ECE documents processed and FAISS index ready in the `faiss_index/` folder.

6. **Run the application**

   ```bash
   streamlit run ece_bot.py
   ```
