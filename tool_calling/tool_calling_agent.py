from dotenv import load_dotenv
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

@tool
def multiply(x: int, y:int) -> int:
    """Multiplies two numbers."""
    return x * y


if __name__ == "__main__":
    tools = [TavilySearch(), multiply]
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that can use tools."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    # llm = ChatOllama(model='gemma3:1b', temperature=0)
    llm = ChatOpenAI(model='gpt-4o')
    agent = create_tool_calling_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    res = agent_executor.invoke({"input": "WHAT's the weather in Dubai right now compare it with weather in India"})
    print(res)