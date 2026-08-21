from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel

class UserMessages(BaseModel):
    messages:list[BaseMessage,add_messages]