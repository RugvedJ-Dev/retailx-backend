from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated,TypedDict
from typing import Any, Dict, Optional
from pydantic import BaseModel

class UserMessages(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]


class AgentRequest(BaseModel):
    inputs: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None