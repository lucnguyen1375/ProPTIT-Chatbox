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
import asyncio
from time import sleep
import openai
logger = logging.getLogger(__name__)

class RAG_Agent(BaseAgent):
    answer_generator : LlmAgent
    general_agent: LlmAgent
    router_agent: LlmAgent
    rewrite_agent: LlmAgent
    def vector_query(tool_context: ToolContext, query_vector: list)-> list:
        logger.info('VECTOR QUERY FUNCTION CALLED')
        collection_name = 'information'
        top_k = 10
        client = MongoClient(os.getenv("MONGODB_URI"))
        db = client.get_database("vector_db")
        collection = db[collection_name]
        if query_vector is None:
            return {"retrival-status": "failed", "error": "No embedded query found in state."}
        
        results = collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",  # tên index bạn đã tạo
                    "queryVector": query_vector,
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": top_k,
                }
            }
        ])
        docs = []
        for doc in results:
            # print(doc['title'])
            # print(doc['information'])
            docs.append(doc['information'])
        return docs

    def embedding(tool_context: ToolContext, text : str) -> list:
        logger.info('EMBEDDING FUNCTION CALLED')
        client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        embedded_query = response.data[0].embedding
        return embedded_query
    
    def __init__(
        self, 
        name: str, 
        router_agent: LlmAgent,
        answer_generator: LlmAgent,
        general_agent: LlmAgent,
        rewrite_agent: LlmAgent
    ):
        super().__init__(
            name=name,
            router_agent=router_agent,
            answer_generator=answer_generator,
            general_agent=general_agent   ,
            rewrite_agent=rewrite_agent          
        )
        
        
    @override
    async def _run_async_impl(self, ctx : InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"RAG Agent {self.name} started.")
        logger.info(f"Session state at start: {ctx.session.state}")
        user_input = ctx.session.state.get("user_input", "")
        route_decision = "general"  # Mặc định là general nếu không có quyết định nào được đưa ra
        async for event in self.router_agent.run_async(ctx):
            if event.is_final_response() and event.content and event.content.parts:
                route_decision = event.content.parts[0].text.strip().lower()
                logger.info(f"Route decision: {route_decision}")
    
        
        
        if route_decision == "rag":
            rewritten_query = ""
            async for event in self.rewrite_agent.run_async(ctx):
                if event.is_final_response() and event.content and event.content.parts:
                    rewritten_query = event.content.parts[0].text.strip()
                    logger.info(f"Rewritten query: {rewritten_query}")
                    
            # embed
            embedded_query = self.embedding(text = rewritten_query)
            # logger.info(f"Embedded query: {embedded_query}")
            ctx.session.state['embedded_query'] = embedded_query
            
            # retrieve
            retrieved_docs = self.vector_query(query_vector = embedded_query)
            logger.info(f"Retrieved documents: {retrieved_docs}")
            ctx.session.state['retrieved_docs'] = retrieved_docs
            
            # rerank
            
            
            async for event in self.answer_generator.run_async(ctx):
                if (event.is_final_response() and event.content and event.content.parts):
                    logger.info(f"Final response from answer generator: {event.content.parts[0].text}")
                yield event
        elif route_decision == "general":
            async for event in self.general_agent.run_async(ctx):
                if (event.is_final_response() and event.content and event.content.parts):
                    logger.info(f"Final response from general agent: {event.content.parts[0].text}")
                yield event