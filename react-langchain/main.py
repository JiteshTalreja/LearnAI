from typing import Union, List

from dotenv import load_dotenv
from langchain_classic.agents.format_scratchpad import format_log_to_str
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentFinish, AgentAction
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool, render_text_description, Tool
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from callbacks import AgentCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the given text by characters."""
    print(f"Calculating length of text: {text}")
    text = text.strip("\n").strip('"')
    return len(text)


def find_tool_by_name(tools: List[Tool], name: str) -> Tool:
    for tool in tools:
        if tool.name == name:
            return tool
    raise ValueError(f"Tool with name {name} not found.")


if __name__ == "__main__":
    print("hello ReAct LangChain!")
    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=" ".join([tool.name for tool in tools]),
    )

    # llm = ChatOpenAI(temperature=0, model_kwargs={"stop":["\nObservation:"]}, model="gpt-4o", callbacks=[AgentCallbackHandler()])
    llm = ChatOllama(temperature=1, stop=["\nObservation:", "observation"], model="gemma3:1b", callbacks=[AgentCallbackHandler()])
    intermediate_step = []

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )
    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {"input": "What is the length of the Dog?", "agent_scratchpad": intermediate_step}
        )
        print(agent_step)

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input
            observation = tool_to_use.invoke(tool_input)
            print(f"Observation: {observation}")
            intermediate_step.append((agent_step, str(observation)))

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)
