from langchain.agents import Tool
from langchain.agents.agent import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from vivo_demo import make_model, weather

# 1. ReAct 에이전트를 위한 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_template(
    """당신은 유능한 기상학자입니다. 필요한 경우 다음 도구를 사용할 수 있습니다:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {question}
Thought:{agent_scratchpad}"""
)


def answer_city(city):
    # 2. 모델과 도구 준비
    llm = make_model(city)
    weather_tool = Tool(name="weather", func=weather.run,
                        description="도시 이름을 받아 현재 날씨를 알려 주는 도구")
    tools = [weather_tool]

    # 3. ReAct 에이전트 생성 및 실행기 설정
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

    # 4. 에이전트 실행
    result = agent_executor.invoke({"question": city})
    return result["output"]


print(answer_city("Seoul"))
