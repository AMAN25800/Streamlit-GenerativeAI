# import streamlit as st
# import os
# import time
# from dotenv import load_dotenv

# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain

# # Load environment variables
# load_dotenv()
# groq_api_key = os.environ['GROQ_API_KEY']

# # Set Ollama remote URL
# OLLAMA_BASE_URL = "http://3.111.226.34:11434"
# os.environ["OLLAMA_BASE_URL"] = OLLAMA_BASE_URL

# # Title
# st.title("🩺 AI Medical Assistant - Symptom to Diagnosis & Treatment")

# # Initialize vector store once
# # Initialize vector store once
# if "vectors" not in st.session_state:
#     with st.spinner("⏳ Loading knowledge base... (First time only)"):
#         embeddings = OllamaEmbeddings(
#             model="nous-hermes2",
#             base_url=OLLAMA_BASE_URL
#         )

#         if os.path.exists("faiss_index/index.faiss"):
#             st.session_state.vectors = FAISS.load_local(
#                 "faiss_index",
#                 embeddings,
#                 allow_dangerous_deserialization=True
#             )
#         else:
#             loader = WebBaseLoader("https://www.geeksforgeeks.org/artificial-intelligence/aiml-introduction/")
#             docs = loader.load()

#             splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
#             split_docs = splitter.split_documents(docs[:30])

#             st.session_state.vectors = FAISS.from_documents(split_docs, embeddings)
#             st.session_state.vectors.save_local("faiss_index")


# # Set up the LLM
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# # Medical prompt template
# prompt_template = ChatPromptTemplate.from_template(
#     """
# You are an experienced medical doctor. Based on the context and user symptoms,
# provide the most likely diagnosis and recommend suitable treatment or medication.

# <context>
# {context}
# </context>

# Symptoms: {input}

# Respond in this format:
# 1. **Possible Diagnosis**
# 2. **Suggested Medicines or Remedies**
# 3. **Note** (if further action or tests are needed)
# """
# )

# # Create the RAG chain
# document_chain = create_stuff_documents_chain(llm, prompt_template)
# retriever = st.session_state.vectors.as_retriever()
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

# # Input from user
# user_input = st.text_input("🤒 Describe your symptoms (e.g., fever, headache, cough):")

# # Output
# if user_input:
#     start = time.time()
#     response = retrieval_chain.invoke({"input": user_input})
#     duration = round(time.time() - start, 2)

#     st.success(f"✅ Response generated in {duration} seconds")
#     st.markdown("### 🩺 AI Diagnosis & Treatment")
#     answer = response.get("answer", "")

#     # If answer is too vague, fallback to bigger LLM
#     if "no mention" in answer.lower() or len(answer.strip()) < 20:
#         st.info("🔄 Refining with more powerful model...")
#         llm2 = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192", temperature=0.3)
#         answer = llm2.invoke(user_input)
#         st.markdown(answer)
#     else:
#         st.markdown(answer)
import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from gtts import gTTS
import base64

# Load environment variables
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

# Set Ollama remote URL
OLLAMA_BASE_URL = "http://3.111.226.34:11434"
os.environ["OLLAMA_BASE_URL"] = OLLAMA_BASE_URL

# Title
st.title("🩺 AI Medical Assistant - Symptom to Diagnosis & Treatment (Text + Voice)")

# Initialize vector store once
if "vectors" not in st.session_state:
    with st.spinner("⏳ Loading knowledge base... (First time only)"):
        embeddings = OllamaEmbeddings(
            model="nous-hermes2",
            base_url=OLLAMA_BASE_URL
        )

        if os.path.exists("faiss_index/index.faiss"):
            st.session_state.vectors = FAISS.load_local(
                "faiss_index",
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            loader = WebBaseLoader("https://www.geeksforgeeks.org/artificial-intelligence/aiml-introduction/")
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            split_docs = splitter.split_documents(docs[:30])

            st.session_state.vectors = FAISS.from_documents(split_docs, embeddings)
            st.session_state.vectors.save_local("faiss_index")

# Set up the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# Prompt Template
prompt_template = ChatPromptTemplate.from_template(
    """
You are an experienced and cautious medical doctor. Based on the context and user symptoms,
provide the most likely diagnosis and recommend **only safe, general remedies** like hydration, rest, or consulting a real doctor.

⚠️ Do NOT suggest specific medications unless they are universally safe and well-known (like paracetamol for fever).
Avoid suggesting antibiotics, controlled substances, or prescription-only drugs.

<context>
{context}
</context>

Symptoms: {input}

Respond in this format:
1. **Possible Diagnosis**
2. **Safe Remedies or Actions** (no strong or specific medications unless clearly essential)
3. **Note** (always advise seeing a real doctor for confirmation)
"""
)


# Create Retrieval QA Chain
document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Input from user
user_input = st.text_input("🤒 Describe your symptoms (e.g., fever, headache, cough):")

def generate_voice(text, filename="response.mp3"):
    tts = gTTS(text)
    tts.save(filename)
    with open(filename, "rb") as audio_file:
        audio_bytes = audio_file.read()
        b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio controls autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """
        return audio_html

# Output
if user_input:
    start = time.time()
    response = retrieval_chain.invoke({"input": user_input})
    duration = round(time.time() - start, 2)

    st.success(f"✅ Response generated in {duration} seconds")
    st.markdown("### 🩺 AI Diagnosis & Treatment")
    answer = response.get("answer", "")

    # Fallback to better model if response is poor
    if "no mention" in answer.lower() or len(answer.strip()) < 20:
        st.info("🔄 Refining with more powerful model...")
        llm2 = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192", temperature=0.3)
        answer = llm2.invoke(user_input)

    st.markdown(answer)

    # Generate and play voice
    st.markdown("### 🔊 Voice Output")
    audio_html = generate_voice(answer)
    st.markdown(audio_html, unsafe_allow_html=True)
