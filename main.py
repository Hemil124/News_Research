import os
import streamlit as st
import time

# from google.api_core.exceptions import ResourceExhausted
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from secret_key import geminiapi_key


os.environ['google_api_key'] = geminiapi_key

st.title("PopatlalBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

main_placeholder = st.empty()
query = main_placeholder.text_input("Question: ")
process_url_clicked = st.button("Process URLs")

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=0.7,
    google_api_key=st.secrets["geminiapi_key"]
)
if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key = st.secrets["geminiapi_key"]
        # google_api_key = geminiapi_key
    )
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    vectorstore_openai.save_local("faiss_index")

if query:
    if os.path.exists("faiss_index"):
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            # google_api_key=geminiapi_key
        )
        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
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
        # except ResourceExhausted:
        except Exception as e:
            st.error("🚫 API quota exceeded. Please check your Gemini API usage or try again later.")


