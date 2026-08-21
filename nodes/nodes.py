from classes.classes import UserMessages
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(model="openai/gpt-oss-20b",temperature=0)

def chat_node(state: UserMessages):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}    
