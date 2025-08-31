import os 
from dotenv import load_dotenv
load_dotenv()

from google.genai.types import Content, Part
from google.adk.agents import BaseAgent, LlmAgent
from typing_extensions import override
from google.adk.agents.invocation_context import InvocationContext
from google.adk.tools import ToolContext
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from typing import AsyncGenerator
from google.genai import types
from pymongo import MongoClient
from google import genai
import logging
from typing_extensions import override
from agents.rag_agent import RAG_Agent
# from agents.root_agent import Root_Agent
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
APP_NAME = "story_app"
USER_ID = "12345"
SESSION_ID = "123344"   
GEMINI_2_FLASH = "gemini-1.5-flash"

general_agent = LlmAgent(
    name="ChitChat_Agent",
    model=GEMINI_2_FLASH,
    instruction="""
    You are a friendly AI assistant. Engage in casual conversation with the user. user_input is stored in {{user_input}}. You can use {{last_conversation}} to have a more friendly answer",
    You should use Vietnamese in your answers, except when the user asks you to use English.
    """
)



answer_generator = LlmAgent(
    name="Answer_Generator",
    model=GEMINI_2_FLASH,
    instruction="""You are a member of ProPTIT,
    Your task is to answer the question based on the provided context.
    Your must use the provided context from the {{retrieved_docs}} to answer the question.
    Don't use any other information.
    If you can't find the answer in the context, say "I don't know".
    Note: The context is provided in Vietnamese, so please answer in Vietnamese as well, convert to pretty format.
    You can use {{last_conversation}} to have a more friendly answer
    """,
    output_key = "response"
)

router_agent = LlmAgent(
    name = "Router_Agent",
    model = GEMINI_2_FLASH,
    instruction = """You are an agent that decides which sub-agent to use based on the user's query stored in {{user_input}}.
    You have two sub-agents:
    1. RAG_Agent: Use this agent to answer questions that require specific information about ProPTIT. This agent retrieves relevant documents from a knowledge base to provide accurate answers.
    2. General_Agent: Use this agent for casual conversation and general questions not related to ProPTIT.
    You need to read input in {{user_input}} and last conversation in {{last_conversation}} to decide which agent is more suitable to answer the question.
    Your output must be either 'rag' or 'general' to indicate which agent to use.
    """,
    output_key = "route_decision"
)
rag_agent = RAG_Agent(
    name = "RAG_Agent", 
    router_agent=router_agent,
    answer_generator=answer_generator,
    general_agent=general_agent
)   

    
INITIAL_SESSION_STATE = {}

async def setup_session_and_runner():
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID, state=INITIAL_SESSION_STATE)
    logger.info(f"Initial session state: {session.state}")
    runner = Runner(
        agent=rag_agent, # Pass the custom orchestrator agent
        app_name=APP_NAME,
        session_service=session_service
    )
    return session_service, runner

async def call_agent_async(user_input: str, last_conversation: dict = None):
    print('---------------------------------------call_agent_async function called-----------------------------------------------------')
    INITIAL_SESSION_STATE['user_input'] = user_input
    INITIAL_SESSION_STATE['last_conversation'] = ""
    
    if last_conversation:
        INITIAL_SESSION_STATE['last_conversation'] = last_conversation
    session_service, runner = await setup_session_and_runner()
    
    content = types.Content(
        role="user",
        parts=[types.Part(text=user_input)]
    )
    events = runner.run_async(
        user_id=USER_ID, 
        session_id=SESSION_ID, 
        new_message=content
    )
    
    final_response = "No response generated."
    
    async for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text
            logger.info(f"Final response received: {final_response}")

    current_session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    logger.info(f"Final session state: ")
    for key, value in current_session.state.items():
        logger.info(f"{key}: {value}")
    
    return final_response


if __name__ == "__main__":
    asyncio.run(call_agent_async("CLB Lập trình ProPTIT là gì?"))