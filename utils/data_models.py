import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

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
    last_response: Optional[AgentResponse] = None

class LongMemRequest(BaseModel):
    # 用户唯一标识
    user_id: str
    # 写入的内容
    memory_info: str
