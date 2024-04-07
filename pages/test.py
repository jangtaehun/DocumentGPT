import streamlit as st
import json
import os
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseOutputParser
from langchain.prompts import PromptTemplate

from prompt import function_call


st.set_page_config(
    page_title="QuizGPT",
    page_icon="💷",
)
st.title("QuizGPT")


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


with st.sidebar:
    openaikey = None
    openaikey = st.text_input("Your OpenAI API key: ", type="password")
    os.environ["OPENAI_API_KEY"] = openaikey

    c = st.container()
    c.link_button("git hub", url="https://github.com/jangtaehun/DocumentGPT")


if openaikey:
    llm = ChatOpenAI(temperature=0.1).bind(
        function_call={"name": "create_quiz"}, functions=[function_call.function]
    )

    #########

    prompt = PromptTemplate.from_template(
        "{city}에 대해 문제를 5개 이상 만들어주세요. 각 문제는 4개의 선택지를 가지고 있습니다. 하나는 정답이고 세 개는 오답입니다."
    )
    chain = prompt | llm
    response = chain.invoke({"city": "수원"})

    response = response.additional_kwargs["function_call"]["arguments"]

    #########

    # embed (X)
    @st.cache_data(show_spinner="Loading file...")
    def split_file(file):
        file_content = file.read()
        file_path = f"files/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader("./files/chapter_one.txt")
        docs = loader.load_and_split(text_splitter=splitter)
        return docs

    # @st.cache_data(show_spinner="Making quiz...")
    # def run_quiz_chain(_docs, topic):
    #     chain = (
    #         {"context": hard_questions_chain} | hard_formatting_chain | output_parser
    #     )
    #     return chain.invoke(_docs)

    with st.sidebar:
        select_custom = st.selectbox(
            "난이도",
            (
                "쉬움",
                "어려움",
            ),
            key="12",
        )

    docs = None
    topic = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )

    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt or .pdf file", type=["docx", "txt", "pdf"]
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")

    if not docs:
        st.markdown(
            """
        QuizeGPT는 Wikipedia와 사용자가 제공한 파일을 이용해 퀴즈를 만들어 제공합니다.
        
        사용을 원하신다면 파일을 업로드하거나 Wikipedia을 선택해 검색어를 입력해주세요.
        """
        )
    else:
        # response = run_quiz_chain(docs, topic if topic else file.name)
        response = chain.invoke({"city": "수원"})
        response = response.additional_kwargs["function_call"]["arguments"]

        st.write(response)
        # count = 0
        # count_score = 0

        # with st.form("questions_form"):
        #     for question in response["questions"]:
        #         st.write(question["질문"])

        #         value = st.radio(
        #             "Select an option.",
        #             [선택지["선택지"] for 선택지 in question["선택지s"]],
        #             index=None,
        #             key=count,
        #         )
        #         count += 1
        #         if {"선택지": value, "정답": True} in question["선택지s"]:
        #             st.success("정답")
        #             count_score += 1
        #         elif value is not None:
        #             st.error("오답")
        #     button = st.form_submit_button()

        if count_score == count:
            st.balloons()
else:
    st.markdown("OPENAI_API_KEY를 입력해주세요")
