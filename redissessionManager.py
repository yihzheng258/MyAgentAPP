import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
from pydantic import BaseModel, Field
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uuid
from langgraph.types import interrupt, Command
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
import uvicorn
from contextlib import asynccontextmanager
import redis.asyncio as redis
import json
from datetime import timedelta, datetime
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from utils.config import Config
from utils.llms import get_llm
from utils.tools import get_tools


class RedisSessionManager:
    def __init__(self, redis_host: str, redis_port: int, redis_db: int, session_timeout: int):
        # 创建 Redis 客户端连接
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        # 设置默认会话过期时间（秒）
        self.session_timeout = session_timeout

    # 关闭 Redis 连接
    async def close(self):
        await self.redis_client.close()
    
    
    # 创建新会话
    async def create_session(self, user_id: str, session_id: Optional[str] = None, status: str = "active",
                            last_query: Optional[str] = None, last_response: Optional['AgentResponse'] = None,
                            last_updated: Optional[float] = None, ttl: Optional[int] = None) -> str:
        if not session_id:
            session_id = str(uuid.uuid4())
        
        if last_updated is None:
            last_updated = str(timedelta(seconds=0))
            
        effective_ttl = ttl if ttl is not None else self.session_timeout
        
        session_data = {
            "session_id": session_id,
            "status": status,
            "last_query": last_query,
            "last_response": last_response.model_dump() if last_response else None,
            "last_query": last_query,
            "last_updated": last_updated
        }
        
        await self.redis_client.set(
            f"session:{user_id}:{session_id}",
            json.dumps(session_data),
            ex=effective_ttl
        )
        
        await self.redis_client.sadd(f"user_sessions:{user_id}", session_id)
        
        return session_id
    
    async def update_session(self, user_id: str, session_id: str, status: Optional[str] = None,
                        last_query: Optional[str] = None, last_response: Optional['AgentResponse'] = None,
                        last_updated: Optional[float] = None, ttl: Optional[int] = None) -> bool:
        if await self.redis_client.exists(f"session:{user_id}:{session_id}"):
            current_data = await self.redis_client.get(f"session:{user_id}:{session_id}")
            if not current_data:
                return False

            if status is not None:
                current_data["status"] = status
            if last_response is not None:
                if isinstance(last_response, BaseModel):
                    current_data["last_response"] = last_response.model_dump()
                else:
                    current_data["last_response"] = last_response
            if last_query is not None:
                current_data["last_query"] = last_query
            if last_updated is not None:
                current_data["last_updated"] = last_updated       
                
            effective_ttl = ttl if ttl is not None else self.session_timeout
            
            await self.redis_client.set(
                f"session:{user_id}:{session_id}",
                json.dumps(current_data),
                ex=effective_ttl
            )
            return True
        else:
            return False
        
        
    async def get_user_active_session_id(self, user_id: str) -> str | None:
        await self.cleanup_user_sessions(user_id)
        
        session_ids = await self.redis_client.smembers(f"user_sessions:{user_id}")
        
        latest_session_id = None
        latest_update_time = -1
        
        for session_id in session_ids:
            session = await self.get_session(user_id, session_id)
            if not session:
                continue
            
            last_updated = session.get("last_updated", 0)
            if last_updated and last_updated > latest_update_time:
                latest_update_time = last_updated
                latest_session_id = session_id
                
        return latest_session_id
    

    
    async def get_session(self, user_id: str, session_id: str) -> Optional[dict]:
        session_data = await self.redis_client.get(f"session:{user_id}:{session_id}")
        if not session_data:
            return None
    
        session = json.loads(session_data)

        if session and "last_response" in session:
            if session["last_response"] is not None:
                session["last_response"] = AgentResponse(**session["last_response"])
                
        return session
    
    
    
    async def get_session_count(self) -> int:
        await self.cleanup_all_sessions()
        count = 0
        user_keys = await self.redis_client.keys("user_sessions:*")
        
        for user_key in user_keys:
            session_count = await self.redis_client.scard(user_key)
            count += session_count

        return count
        
   
    async def get_all_users_session_ids(self, user_id: str) -> List[str]:
        #清理无效对话id
        await self.cleanup_user_sessions(user_id)
        session_ids = await self.redis_client.smembers(f"user_sessions:{user_id}")
        
        return list(session_ids)
    
    async def get_all_users_session_ids(self) -> Dict[str, List[str]]:
        await self.cleanup_all_sessions()
        result = {}
        
        user_keys = await self.redis_client.keys("user_sessions:*")
        for user_key in user_keys:
            user_id = user_key.split(":", 1)[1]
            session_ids = await self.redis_client.smembers(user_key)
            result[user_id] = list(session_ids)
        
        return result
    
    async def get_all_user_sessions(self, user_id: str) -> List[dict]:
        await self.cleanup_user_sessions(user_id)
        session_ids = await self.redis_client.smembers(f"user_sessions:{user_id}")
        
        sessions = []
        for session_id in session_ids:
            session = await self.get_session(user_id, session_id)
            if session:
                sessions.append(session)
        
        return sessions
    
    async def user_id_exists(self, user_id: str) -> bool:
        await self.cleanup_user_sessions(user_id)
        user_key = f"user_sessions:{user_id}"
        return await self.redis_client.exists(user_key) > 0
    
    async def session_id_exists(self, user_id: str, session_id: str) -> bool:
        await self.cleanup_user_sessions(user_id)
        return (await self.redis_client.exists(f"session:{user_id}:{session_id}")) > 0


    
    async def cleanup_all_sessions(self) -> None:
        user_keys = await self.redis_client.keys("user_sessions:*")
        for user_key in user_keys:
            user_id = user_key.split(":", 1)[1]
            await self.cleanup_user_sessions(user_id)
            
   
    async def cleanup_user_sessions(self, user_id: str) -> None:
        session_ids = await self.redis_client.smembers(f"user_sessions:{user_id}")

        for session_id in session_ids:
            if not await self.redis_client.exists(f"session:{user_id}:{session_id}"):
                # 如果会话键已过期或不存在，从集合中移除 session_id
                await self.redis_client.srem(f"user_sessions:{user_id}", session_id)
                logger.info(f"Removed expired session_id {session_id} for user {user_id}")
        
        if not await self.redis_client.scard(f"user_sessions:{user_id}"):
            await self.redis_client.delete(f"user_sessions:{user_id}")
            logger.info(f"Deleted empty user_sessions collection for user {user_id}")

    
    async def delete_session(self, user_id: str, session_id: str) -> bool:
        await self.redis_client.srem(f"user_sessions:{user_id}", session_id)
        deleted = await self.redis_client.delete(f"session:{user_id}:{session_id}")
        return deleted > 0
        
