from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from .system_prompt import SYSTEM_PROMPT
from .tools import TOOLS


def build_agent():
    model = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
    checkpointer = MemorySaver()
    return create_agent(
        model,
        tools=TOOLS,
        checkpointer=checkpointer,
        system_prompt=SYSTEM_PROMPT,
    )
