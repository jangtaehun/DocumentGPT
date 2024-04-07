from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

function = {
    "name": "create_quiz",
    "description": "질문과 선택지의 리스트를 가져와 퀴즈를 생성하는 함수",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "질문": {
                            "type": "string",
                        },
                        "선택지s": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "선택지": {
                                        "type": "string",
                                    },
                                    "정답": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["선택지", "정답"],
                            },
                        },
                    },
                    "required": ["질문", "선택지s"],
                },
            }
        },
        "required": ["lon", "lat"],
    },
}

# llm = ChatOpenAI(temperature=0.1).bind(
#     function_call={"name": "create_quiz"},
#     # function_call= "auto",
#     functions=[function],
# )

# prompt = PromptTemplate.from_template(
#     "{city}에 대해 문제를 5개 이상 만들어주세요. 각 문제는 4개의 선택지를 가지고 있습니다. 하나는 정답이고 세 개는 오답입니다."
# )
# chain = prompt | llm
# response = chain.invoke({"city": "수원"})

# response = response.additional_kwargs["function_call"]["arguments"]

# response
