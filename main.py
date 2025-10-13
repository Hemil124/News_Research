import os
import streamlit as st
import time
import requests
from bs4 import BeautifulSoup

from google.api_core.exceptions import ResourceExhausted
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from secret_key import geminiapi_key

os.environ['google_api_key'] = geminiapi_key

st.title("InfoDigger — AI Research & Insight Tool")
st.sidebar.title("Web Content URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

main_placeholder = st.empty()
query_placeholder = st.empty()
query = query_placeholder.text_input("Question: ")
process_url_clicked = st.button("Process URLs")

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.7,
    google_api_key=st.secrets["geminiapi_key"] # it use for when deploy
)

# --- Robust fetch_html function ---
def fetch_html(url, timeout=10):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
    }
    try:
        res = requests.get(url, headers=headers, timeout=timeout)
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.warning(f"⚠️ Could not load URL: {url}\nReason: {e}")
        return None
    soup = BeautifulSoup(res.text, "html.parser")
    return soup.get_text(separator="\n")

if process_url_clicked:
    main_placeholder.text("Data Loading...Started...✅✅✅")

    # Load each URL robustly
    data = []
    for url in urls:
        if url.strip() == "":
            continue
        text = fetch_html(url)
        if text:
            # Convert to LangChain Document
            data.append(Document(page_content=text, metadata={"source": url}))

    if not data:
        st.error("❌ No valid data loaded from the URLs.")
    else:
        # Split data
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        docs = text_splitter.split_documents(data)

        # Create embeddings & FAISS index
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            # google_api_key=geminiapi_key,
            google_api_key = st.secrets["geminiapi_key"] # it use for when deploy
        )
        vectorstore = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)
        vectorstore.save_local("faiss_index")

if query:
    if os.path.exists("faiss_index"):
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            # google_api_key=geminiapi_key,
            google_api_key = st.secrets["geminiapi_key"] # it use for when deploy
        )
        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )
        try:
            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.write(result["answer"])
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
        except ResourceExhausted:
            st.error("🚫 API quota exceeded. Please check your Gemini API usage or try again later.")
