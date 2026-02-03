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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.handlers = []  # 清空默认处理器
# 使用ConcurrentRotatingFileHandler
handler = ConcurrentRotatingFileHandler(
    Config.LOG_FILE,
    maxBytes = Config.MAX_BYTES,
    backupCount = Config.BACKUP_COUNT
)
# 设置处理器级别为DEBUG
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)


#定义数据模型

# 前端传给后端的请求数据
class AgentRequest(BaseModel):
    session_id: str
    user_id: str
    query: str
    system_message: Optional[str] = "你会使用工具来帮助用户。如果工具使用被拒绝，请提示用户。"

#后端相应数据
class AgentResponse(BaseModel):
    session_id: str
    # interrupted, completed, error
    status: str
    timestamp: float = Field(default_factory=lambda: time.time())
    # error时的提示消息
    message: Optional[str] = None
    # completed时的结果消息
    result: Optional[Dict[str, Any]] = None
    # interrupted时的中断消息
    interrupt_data: Optional[Dict[str, Any]] = None

#前端给后端的中断恢复请求数据
class InterruptResponse(BaseModel):
    user_id: str
    session_id: str
    # accept, reject, edit, response
    response_type: str  
    # for edit
    args: Optional[Dict[str, Any]] = None
    
class SystemInfoResponse(BaseModel):
    sessions_count: int
    active_users: Optional[Dict[str, Any]] = None

class SessionInfoResponse(BaseModel):
    session_ids: List[str]

class ActiveSessionInfoResponse(BaseModel):
    # 最近一次更新的会话ID
    active_session_id: str

class SessionStatusResponse(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    # not_found, idle, running, interrupted, completed, error
    status: str
    # error message
    message: Optional[str] = None
    last_query: Optional[str] = None
    last_updated: Optional[float] = None
    last_response: Optional[str] = None


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
        


class SystemInfoResponse(BaseModel):
    session_count: int 
    active_users: Optional[Dict[str, Any]] = None


# 生命周期函数 app应用初始化函数
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        #初始化redis会话管理器
        app.state.session_manager = RedisSessionManager(
            Config.REDIS_HOST,
            Config.REDIS_PORT,
            Config.REDIS_DB,
            Config.SESSION_TIMEOUT
        )
        logger.info("Redis初始化成功")
        
        llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)
        logger.info("Chat模型初始化成功")
        
    
    except Exception as e:
        logger.error(f"应用初始化失败: {e}")
        raise RuntimeError(f"服务初始化失败: {str(e)}")
    
    

@app.get("/system/info", response_model=SystemInfoResponse)
async def get_system_info():
    logger.info(f"调用/system/info接口，获取当前系统内全部的会话状态信息")

    response = SystemInfoResponse(
        sessions_count=await app.state.session_manager.get_session_count(),
        active_users=await app.state.session_manager.get_all_users_session_ids()
    )
    logger.info(f"返回当前系统状态信息:{response}")
    return response



   