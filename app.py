import streamlit as st
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks import StreamingStdOutCallbackHandler
import os

CACHE_DIR = "./.cache"
st.set_page_config(
    page_title="Streamlit",
    page_icon="🤖",
)

@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()

    file_path = os.path.join(CACHE_DIR, "files", file.name)
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)

    loader = UnstructuredFileLoader(file_path)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings()
    cache_dir = LocalFileStore(os.path.join(CACHE_DIR, "embeddings", file.name))
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    # Retrieve context
    retriever = vectorstore.as_retriever()
    return retriever

api_key = None
with st.sidebar:
    api_key = st.text_input("Write down your OpenAI key", placeholder="sk-proj-NDE*********")

    file = st.file_uploader(
        "Upload a .txt .pdf .docx or .md file",
        type=["pdf", "txt", "docx", "md"],
    )

    st.write("<a href='https://github.com/kyong-dev/gpt-challenge-streamlit'>https://github.com/kyong-dev/gpt-challenge-streamlit</a>", unsafe_allow_html=True)

if api_key and file:
    llm = ChatOpenAI(
        model_name="gpt-4o",
        streaming=True,
        api_key=api_key,
        temperature=0.1,
        callbacks=[
            StreamingStdOutCallbackHandler(),
        ],
    )

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=120,
        return_messages=True,
    )

    retriever = embed_file(file)

    # Define prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer questions using only the following context. "
                "If you don't know the answer just say you don't know, don't make it up:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    # Define memory loader
    def load_memory(_):
        return memory.load_memory_variables({})["history"]
    
    chain = (
    {
        "context": retriever,
        "history": load_memory,
        "question": RunnablePassthrough(),
    }
        | prompt
        | llm
    )

    # Define inputs and run the chain
    inputs = [
        "Is Aaronson guilty?",
        "What message did he write in the table?",
        "Who is Julia?",
    ]

    for input in inputs:
        result = chain.invoke({"context": retriever, "question": input, "history": load_memory({})})

        # Save to memory
        memory.save_context(
            {"input": input},
            {"output": result["content"]},
        )
        
        # Print result
        st.write(result["content"])