from classes.classes import UserMessages
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from config.mcp import load_mcp_tools

load_dotenv()

llm = ChatGroq(model="qwen/qwen3.6-27b",temperature=0,streaming=True)

mcp_tools = load_mcp_tools()

llm_with_tools = llm.bind_tools(mcp_tools) if mcp_tools else llm

async def chat_node(state: UserMessages):
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}    
