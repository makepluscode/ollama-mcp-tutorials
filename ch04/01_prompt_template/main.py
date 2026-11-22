from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vivo_demo import make_chat_model

# 1. 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_template(
    "당신은 유능한 기상학자입니다. 다음 질문에 답해주세요. <질문>: {question}"
)

# 2. 모델과 출력 파서 설정
llm = make_chat_model()
output_parser = StrOutputParser()

# 3. 프롬프트 → 모델 → 출력 파서 체인 구성
chain = prompt | llm | output_parser


def ask(question):
    return chain.invoke({"question": question})


print(ask("서울 날씨는?"))
