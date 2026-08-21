from langgraph.graph import StateGraph,START,END
from nodes.nodes import chat_node
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.memory import InMemorySaver
from classes.classes import UserMessages,AgentRequest
from langgraph.prebuilt import ToolNode, tools_condition
from nodes.nodes import chat_node
from config.mcp import load_mcp_tools

app = FastAPI()

mcp_tools = load_mcp_tools()

tool_node = ToolNode(mcp_tools) if mcp_tools else None

memory = InMemorySaver()

builder = StateGraph(UserMessages)
builder.add_node("chatNode",chat_node)
builder.add_edge(START,"chatNode")
builder.add_edge("chatNode",END)
if tool_node:
    builder.add_node("tools", tool_node)
    builder.add_conditional_edges("chatNode", tools_condition)
    builder.add_edge("tools", "chatNode")
else:
    builder.add_edge("chatNode", END)

graph = builder.compile(checkpointer=memory)


async def event_generator(inputs, config):
    # Track if an interrupt paused the graph
    has_interrupted = False

    async for event in graph.astream_events(inputs, config, version="v2"):
        kind = event["event"]

        # 1. EXACT PYTHON TOOL START (Fires the moment a tool function begins)
        if kind == "on_tool_start":
            yield f"data: {json.dumps({'event_type': 'tool_start', 'tool_name': event['name'], 'input': event['data'].get('input')})}\n\n"

        # 2. EXACT PYTHON TOOL END (Fires when the tool function finishes)
        elif kind == "on_tool_end":
            yield f"data: {json.dumps({'event_type': 'tool_end', 'tool_name': event['name'], 'output': str(event['data'].get('output'))})}\n\n"

        # 3. REAL-TIME LLM TOKENS (Streams text as it is generated)
        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content and isinstance(chunk.content, str):
                yield f"data: {json.dumps({'event_type': 'token', 'content': chunk.content})}\n\n"

        # 4. NODE COMPLETION (Fires when a graph node finishes its turn)
        elif kind == "on_chain_end" and event["name"] in graph.nodes:
            yield f"data: {json.dumps({'event_type': 'node_complete', 'node': event['name']})}\n\n"

    # 5. CHECK FOR HUMAN-IN-THE-LOOP INTERRUPTS
    # Inspect state right after stream loop completes to see if graph was paused
    state = await graph.aget_state(config)
    if state.next:
        for task in state.tasks:
            if task.interrupts:
                has_interrupted = True
                yield f"data: {json.dumps({
                    'event_type': 'interrupt',
                    'node': task.name,
                    'interrupt_id': task.interrupts[0].id,
                    'value': task.interrupts[0].value
                })}\n\n"

    # 6. STREAM TERMINATION SIGNALS
    if has_interrupted:
        yield f"data: {json.dumps({'event_type': 'stream_paused', 'reason': 'interrupt'})}\n\n"
    else:
        yield f"data: {json.dumps({'event_type': 'stream_end'})}\n\n"


@app.post("/api/stream")
async def stream_agent(request: AgentRequest):
    return StreamingResponse(
        event_generator(request.inputs, request.config), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disables proxy buffering (Nginx/Vercel)
            "Content-Type": "text/event-stream",
        }
    )