from typing import Annotated,TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]