import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import os


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *arg, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *arg, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    # file_path = f"files/{file.name}"
    file_path = f"./.cache/files/{file.name}"
    os.makedirs(f"./..cache/files/", exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(file_content)
    loader = UnstructuredFileLoader(f"files/{file.name}")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    cache_dir = LocalFileStore(f"files/embeddings/{file.name}")

    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


# message, role 저장
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


# message 기록 보이기, 저장X
def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


st.set_page_config(page_title="CodeChallengeGPT", page_icon="📚")


with st.sidebar:

    # openaikey = st.text_input(" Your OpenAI API key: ", type="password")
    st.session_state["api_key"] = st.text_input("Your OpenAI API Key")

    file = st.file_uploader("파일을 업로드해주세요!!", type=["pdf", "txt", "docx"])

    c = st.container()
    c.link_button("git hub", url="https://github.com/jangtaehun/DocumentGPT")


template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    {context}만을 사용해 질문에 답변을 해주세요. 모른다면 모른다고 하세요. 꾸며내지 마세요.
    """,
        ),
        ("human", "{question}"),
    ]
)


st.title("DocumentGPT")


# file이 존재하면 실행
if file and st.session_state["api_key"]:

    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
        openai_api_key=st.session_state["api_key"],
    )

    retriever = embed_file(file)

    send_message("나는 준비됐어 물어봐!!", "ai", save=False)
    paint_history()

    message = st.chat_input("첨부한 파일에 대해 어떤 것이든 물어봐!!")

    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | template
            | llm
        )

        with st.chat_message("ai"):
            response = chain.invoke(message)

else:
    st.session_state["messages"] = []
