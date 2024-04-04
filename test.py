from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models.openai import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import faiss
import streamlit as st
import os


# classes
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(
        self, *args, **kwargs
    ):  # can receive as many arguments and keyword arguments
        self.message_box = st.empty()  # empty widget you can later put things inside of

    def on_llm_end(
        self, *args, **kwargs
    ):  # can receive as many arguments and keyword arguments
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token  # self. because inside of class
        self.message_box.markdown(self.message)


# functions
@st.cache_data(show_spinner=True)
def embed_file(file):
    # create a copy of the file to work with
    file_content = file.read()
    file_content = file.read()
    file_path = f"files/{file.name}"

    # file_path = f"./.cache/files/{file.name}"
    # os.makedirs(f"./..cache/files/", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)

    # load and split the file
    loader = UnstructuredFileLoader(file_path)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    doc = loader.load_and_split(text_splitter=splitter)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    embedder = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embedder, cache_dir)

    # create a vector store
    vectorStore = faiss.FAISS.from_documents(doc, cached_embeddings)

    retriever = vectorStore.as_retriever()
    return retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_message():
    history = []

    for message in st.session_state["messages"]:
        tup = (f"{message['role']}", f"{message['message']}")
        history.append(tup)

    return ChatPromptTemplate.from_messages(history)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], False)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)

    if save:
        st.session_state["messages"].append({"message": message, "role": role})


# streamlit

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄",
)

st.title("Document GPT")

st.markdown(
    """
Use this chatbot to ask questions to an AI about your files!
"""
)

with st.sidebar:
    st.write(
        "This project is brought to you by:\nhttps://github.com/seansanghalee/fullstack_gpt"
    )
    st.session_state["api_key"] = st.text_input("Your OpenAI API Key")
    file = st.file_uploader(
        "Upload a document",
        type=[
            "docx",
            "pdf",
            "txt",
        ],
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
     Use the following pieces of context to answer the user's question.
     If you don't know the answer just say you don't know, don't try to make up an answer.
     ----------------
     {context}
     {history}
     """,
        ),
        ("human", "{question}"),
    ]
)

if file and st.session_state["api_key"]:
    llm = ChatOpenAI(
        temperature=0,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
        api_key=st.session_state["api_key"],
    )

    retriever = embed_file(file)

    send_message("I'm ready, ask me anything!", "ai", False)
    paint_history()

    message = st.chat_input("Ask anything about your file...")

    if message:
        send_message(message, "human")

        chain = (
            RunnableParallel(
                {
                    "context": retriever
                    | RunnableLambda(format_docs),  # docs = retriever.invoke(message)
                    "question": RunnablePassthrough(),
                    "history": load_message(),  # bring all the messages from session_state
                }
            )
            | prompt
            | llm
        )

        with st.chat_message("ai"):
            response = chain.invoke(message)

else:
    st.session_state["messages"] = []
