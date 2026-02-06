import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any, Dict, List, Optional

import uvicorn
from concurrent_log_handler import ConcurrentRotatingFileHandler
from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.messages.utils import trim_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.message import add_messages
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.types import Command
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from typing_extensions import TypedDict

from MyMultiAgentAPP.utils.data_models import (
    ActiveSessionInfoResponse,
    AgentRequest,
    AgentResponse,
    InterruptResponse,
    LongMemRequest,
    SessionInfoResponse,
    SessionStatusResponse,
    SystemInfoResponse,
)
from MyMultiAgentAPP.utils.redis_session_manager import RedisSessionManager
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

TRAVEL_PLAN_KEYWORDS = (
    "旅游",
    "旅行",
    "行程",
    "攻略",
    "景点",
    "路线",
    "出行",
    "travel",
    "itinerary",
)


class MultiAgentState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    user_query: str


def should_generate_travel_plan(user_query: Optional[str]) -> bool:
    if not user_query:
        return False
    normalized_query = user_query.lower()
    return any(keyword in normalized_query for keyword in TRAVEL_PLAN_KEYWORDS)


def build_writer_context(messages: List[AnyMessage]) -> tuple[str, str]:
    core_answer = ""
    tool_notes: List[str] = []

    for message in messages:
        if isinstance(message, ToolMessage):
            tool_name = getattr(message, "name", "tool")
            tool_content = getattr(message, "content", "")
            if tool_content:
                tool_notes.append(f"[{tool_name}] {tool_content}")

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                core_answer = content
                break

    return core_answer, "\n".join(tool_notes)


def trim_history_messages(messages: List[AnyMessage]) -> List[AnyMessage]:
    return trim_messages(
        messages=messages,
        max_tokens=20,
        strategy="last",
        token_counter=len,
        start_on="human",
        allow_partial=False,
    )


async def researcher_llm_node(state: MultiAgentState, config: RunnableConfig) -> Dict[str, Any]:
    llm_with_tools = app.state.query_llm.bind_tools(app.state.tools)
    messages = state.get("messages", [])
    llm_input_messages = trim_history_messages(messages)
    ai_msg = await llm_with_tools.ainvoke(llm_input_messages, config=config)
    return {"messages": [ai_msg]}


def researcher_router(state: MultiAgentState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return END
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "research_tools"
    if should_generate_travel_plan(state.get("user_query")):
        return "writer_agent"
    return END


async def writer_agent_node(state: MultiAgentState, config: RunnableConfig) -> Dict[str, Any]:
    core_answer, tool_notes = build_writer_context(state.get("messages", []))
    writer_prompt = [
        SystemMessage(
            content=(
                "你是Writer Agent，请基于查询Agent和工具检索信息生成旅游计划。"
                "输出应包含：每日行程、交通建议、预算建议、注意事项。"
                "如果信息不足，请明确缺口并给出补充建议。"
            )
        ),
        HumanMessage(
            content=(
                f"用户需求：{state.get('user_query', '')}\n\n"
                f"查询Agent结论：{core_answer or '无'}\n\n"
                f"工具检索信息：\n{tool_notes or '无'}\n\n"
                "请输出一份可执行的中文旅游计划。"
            )
        ),
    ]
    writer_reply = await app.state.writer_llm.ainvoke(writer_prompt, config=config)
    return {"messages": [AIMessage(content=writer_reply.content, name="WriterAgent")]}


def build_multi_agent_graph(checkpointer: AsyncPostgresSaver):
    graph_builder = StateGraph(MultiAgentState)
    graph_builder.add_node("researcher_llm", researcher_llm_node)
    graph_builder.add_node("research_tools", ToolNode(app.state.tools))
    graph_builder.add_node("writer_agent", writer_agent_node)
    graph_builder.add_edge(START, "researcher_llm")
    graph_builder.add_conditional_edges(
        "researcher_llm",
        researcher_router,
        {"research_tools": "research_tools", "writer_agent": "writer_agent", END: END},
    )
    graph_builder.add_edge("research_tools", "researcher_llm")
    graph_builder.add_edge("writer_agent", END)
    return graph_builder.compile(checkpointer=checkpointer)

# 读取指定用户长期记忆中的内容
async def read_long_term_info(user_id: str):
    
    try:
        namespace = ("memories", user_id)
        
        memories = await app.state.store.asearch(namespace, query="")

        if memories is None:
            raise HTTPException(status_code=500, detail="未找到长期记忆信息")
        
        info = " ".join([d.value["data"] for d in memories]) if memories else "无长期记忆信息"
        logger.info(f"成功获取用户ID: {user_id} 的长期记忆，内容长度: {len(info)} 字符")
        
        return {
            "success": True,
            "user_id": user_id,
            "long_term_info": info,
            "message": "成功获取长期记忆信息" if info else "无长期记忆信息"
        }
    
    except Exception as e:
        logger.error(f"获取用户ID: {user_id} 的长期记忆失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取长期记忆失败: {str(e)}")

# 写入指定用户长期记忆内容
async def write_long_term_info(user_id :str, memory_info :str):
    try:
        namespace = ("memories", user_id)
        memory_id = str(uuid.uuid4())
        
        await app.state.store.aput(
            namespace=namespace,
            key=memory_id,
            value={
                "data": memory_info,
            }
        )
        
        logger.info(f"成功写入用户ID: {user_id} 的长期记忆，内容长度: {len(memory_info)} 字符")
        
        return {
            "success": True,
            "memory_id": memory_id,
            "message": "成功写入长期记忆信息"
        }
    
    except Exception as e:
        logger.error(f"写入用户ID: {user_id} 的长期记忆失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"写入长期记忆失败: {str(e)}")
    
    
# 生命周期函数 app应用初始化函数
@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = None
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
        
        async with AsyncConnectionPool(
            conninfo=Config.DB_URL,
            min_size=Config.MIN_SIZE,
            max_size=Config.MAX_SIZE, 
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row
            }
        ) as pool:
            #初始化短期记忆
            app.state.checkpointer = AsyncPostgresSaver(pool)
            await app.state.checkpointer.setup()
            logger.info("短期记忆初始化成功")
            
            #初始化长期记忆
            app.state.store = AsyncPostgresStore(pool)
            await app.state.store.setup()
            logger.info("长期记忆store初始化成功")
            
            tools = await get_tools()
            app.state.tools = tools
            app.state.query_llm = llm_chat
            app.state.writer_llm = llm_chat
            app.state.agent = build_multi_agent_graph(app.state.checkpointer)

            logger.info("MultiAgent初始化成功")

            logger.info("服务完成初始化并启动服务")
            yield
    
    except Exception as e:
        logger.error(f"应用初始化失败: {e}")
        raise RuntimeError(f"服务初始化失败: {str(e)}")
    
    finally:
        # 关闭redis连接
        if hasattr(app.state, "session_manager"):
            await app.state.session_manager.close()
            logger.info("Redis连接已关闭")
        
        
        # 关闭数据库连接池
        if pool is not None:
            await pool.close()
            logger.info("数据库连接池已关闭")


app = FastAPI(
    title="Agent智能体后端API接口服务",
    description="基于LangGraph提供AI Agent服务",
    lifespan=lifespan
)
    
async def parse_messages(messages: List[Any]) -> None:
    print("=== 消息解析结果 ===")
    for idx, msg in enumerate(messages, 1):
        print(f"\n消息 {idx}:")
        # 获取消息类型
        msg_type = msg.__class__.__name__
        print(f"类型: {msg_type}")
        content = getattr(msg, 'content', '')
        print(f"内容: {content if content else '<空>'}")
        # 处理附加信息
        additional_kwargs = getattr(msg, 'additional_kwargs', {})
        if additional_kwargs:
            print("附加信息:")
            for key, value in additional_kwargs.items():
                if key == 'tool_calls' and value:
                    print("  工具调用:")
                    for tool_call in value:
                        print(f"    - ID: {tool_call['id']}")
                        print(f"      函数: {tool_call['function']['name']}")
                        print(f"      参数: {tool_call['function']['arguments']}")
                else:
                    print(f"  {key}: {value}")
        # 处理 ToolMessage 特有字段
        if msg_type == 'ToolMessage':
            tool_name = getattr(msg, 'name', '')
            tool_call_id = getattr(msg, 'tool_call_id', '')
            print(f"工具名称: {tool_name}")
            print(f"工具调用 ID: {tool_call_id}")
        # 处理 AIMessage 的工具调用和元数据
        if msg_type == 'AIMessage':
            tool_calls = getattr(msg, 'tool_calls', [])
            if tool_calls:
                print("工具调用:")
                for tool_call in tool_calls:
                    print(f"  - 名称: {tool_call['name']}")
                    print(f"    参数: {tool_call['args']}")
                    print(f"    ID: {tool_call['id']}")
            # 提取元数据
            metadata = getattr(msg, 'response_metadata', {})
            if metadata:
                print("元数据:")
                token_usage = metadata.get('token_usage', {})
                print(f"  令牌使用: {token_usage}")
                print(f"  模型名称: {metadata.get('model_name', '未知')}")
                print(f"  完成原因: {metadata.get('finish_reason', '未知')}")
        # 打印消息 ID
        msg_id = getattr(msg, 'id', '未知')
        print(f"消息 ID: {msg_id}")
        print("-" * 50)
 
#中断或完成       
async def process_agent_result(
        session_id: str,
        result: Dict[str, Any],
        user_id: Optional[str] = None
) -> AgentResponse:
    response = None
    
    try:
        if "__interrupt__" in result:
            interrupt_data = result["__interrupt__"][0].value
            # 确保中断数据有类型信息
            if "interrupt_type" not in interrupt_data:
                interrupt_data["interrupt_type"] = "unknown"
            # 返回中断信息
            response = AgentResponse(
                session_id=session_id,
                status="interrupted",
                interrupt_data=interrupt_data
            )
            logger.info(f"当前触发工具调用中断:{response}")
        else:
            response = AgentResponse(
                session_id=session_id,
                status="completed",
                result=result
            )
            logger.info(f"最终智能体回复结果:{response}")
    
    except Exception as e:
        response = AgentResponse(
            session_id=session_id,
            status="error",
            message=f"处理智能体结果时出错: {str(e)}"
        )
        logger.error(f"处理智能体结果时出错:{response}")

    exists = await app.state.session_manager.session_id_exists(user_id, session_id)
    if exists:
        # 更新会话状态
        status = response.status
        last_query = None
        last_response = response
        last_updated = time.time()
        ttl = Config.TTL
        await app.state.session_manager.update_session(user_id, session_id, status, last_query, last_response, last_updated, ttl)
        
    return response

#API接口：获取全部会话状态信息
@app.get("/system/info", response_model=SystemInfoResponse)
async def get_system_info():
    logger.info(f"调用/system/info接口，获取当前系统内全部的会话状态信息")

    response = SystemInfoResponse(
        sessions_count=await app.state.session_manager.get_session_count(),
        active_users=await app.state.session_manager.get_all_users_session_ids()
    )
    logger.info(f"返回当前系统状态信息:{response}")
    return response


@app.post("/agent/invoke", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest):
    logger.info(f"调用/agent/invoke接口，用户ID:{request.user_id}，会话ID:{request.session_id}，查询内容:{request.query}")

    user_id = request.user_id
    session_id = request.session_id
    
    long_term_info = await read_long_term_info(user_id)
    
    if long_term_info.get("success", False):
        info = long_term_info.get("long_term_info")
        if info:
            system_message = f"{request.system_message}我的附加信息有:{info}"
            logger.info(f"获取用户偏好配置数据，system_message的信息为:{system_message}")
        else:
            system_message = request.system_message
            logger.info(f"无用户偏好配置信息，system_message的信息为:{system_message}")
    else:
        system_message = request.system_message
        logger.info(f"获取用户偏好配置数据失败，system_message的信息为:{system_message}")        

    exists = await app.state.session_manager.session_id_exists(user_id, session_id)
    if not exists:
        status = "idle"
        last_query = None
        last_response = None
        last_updated = time.time()
        ttl = Config.TTL
        # 创建会话并存储到redis中
        await app.state.session_manager.create_session(user_id, session_id, status, last_query, last_response, last_updated, ttl)

    # 因为要提交query，所以status马上变为running
    status = "running"
    last_query = request.query
    last_response = None
    last_updated = time.time()
    ttl = Config.TTL
    await app.state.session_manager.update_session(user_id, session_id, status, last_query, last_response, last_updated, ttl)

    # 构造智能体输入消息体
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=request.query),
    ]
    
    try: 
        result = await app.state.agent.ainvoke(
            {"messages": messages, "user_query": request.query},
            config={"configurable": {"thread_id": session_id}},
        )
        await parse_messages(result.get("messages", []))
        
        return await process_agent_result(session_id, result, user_id)
        
    except Exception as e:
        # 异常处理
        error_response = AgentResponse(
            session_id=session_id,
            status="error",
            message=f"处理请求时出错: {str(e)}"
        )
        logger.error(f"处理请求时出错: {error_response}")

        # 更新会话状态
        status = "error"
        last_query = None
        last_response = error_response
        last_updated = time.time()
        ttl = Config.TTL
        await app.state.session_manager.update_session(user_id, session_id, status, last_query, last_response, last_updated, ttl)

        return error_response

@app.post("/agent/resume", response_model=AgentResponse)
async def resume_agent(response: InterruptResponse):
    logger.info(f"调用/agent/resume接口，恢复被中断的智能体运行并等待运行完成或再次中断，接受到前端用户请求:{response}")

    user_id = response.user_id
    session_id = response.session_id
    
    #判断当前会话是否存在
    exists = await app.state.session_manager.session_id_exists(user_id, session_id)
    if not exists:
        logger.error(f"status_code=404,用户会话 {user_id}:{session_id} 不存在")
        raise HTTPException(status_code=404, detail=f"用户会话 {user_id}:{session_id} 不存在")

    #如果会话状态不是interrupt，则抛出异常
    session = await app.state.session_manager.get_session(user_id, session_id)
    status = session.get("status")
    if status != "interrupted":
        logger.error(f"status_code=400,会话当前状态为 {status}，无法恢复非中断状态的会话")
        raise HTTPException(status_code=400, detail=f"会话当前状态为 {status}，无法恢复非中断状态的会话")

    # 更新会话状态
    status = "running"
    last_query = None
    last_response = None
    last_updated = time.time()
    ttl = Config.TTL
    await app.state.session_manager.update_session(user_id, session_id, status, last_query, last_response, last_updated, ttl)

    command_data = {
        "type": response.response_type
    }
    
    if response.args:
        command_data["args"] = response.args
    
    try:
        result = await app.state.agent.ainvoke(
            Command(resume=command_data),
            config={"configurable": {"thread_id": session_id}}
        )
        await parse_messages(result.get("messages", []))
        
        return await process_agent_result(session_id, result, user_id)

    except Exception as e:
        # 异常处理
        error_response = AgentResponse(
            session_id=session_id,
            status="error",
            message=f"处理请求时出错: {str(e)}"
        )
        logger.error(f"处理请求时出错: {error_response}")

        # 更新会话状态
        status = "error"
        last_query = None
        last_response = error_response
        last_updated = time.time()
        ttl = Config.TTL
        await app.state.session_manager.update_session(user_id, session_id, status, last_query, last_response, last_updated, ttl)

        return error_response

# API接口:获取指定用户当前会话的状态数据
@app.get("/agent/status/{user_id}/{session_id}", response_model=SessionStatusResponse)
async def get_agent_session_status(user_id: str, session_id: str):
    logger.info(f"调用/agent/status/接口，获取指定用户当前会话的状态数据，接受到前端用户请求:{user_id}:{session_id}")
    
    exists = await app.state.session_manager.session_id_exists(user_id, session_id)
    
    if not exists:
        logger.error(f"status_code=404,用户会话 {user_id}:{session_id} 不存在")
        return SessionStatusResponse(
            user_id=user_id,
            session_id=session_id,
            status="not_found",
            message=f"用户 {user_id}:{session_id} 的会话不存在"
        )
    
    #若会话存在
    session = await app.state.session_manager.get_session(user_id, session_id)
    response = SessionStatusResponse(
        user_id=user_id,
        session_id=session_id,
        status=session.get("status"),
        last_query=session.get("last_query"),
        last_updated=session.get("last_updated"),
        last_response=session.get("last_response")
    )
    logger.info(f"返回当前用户的会话状态:{response}")
    return response

# API接口:获取指定用户当前最近一次更新的会话ID
@app.get("/agent/active/sessionid/{user_id}", response_model=ActiveSessionInfoResponse)
async def get_agent_active_sessionid(user_id: str):
    logger.info(f"调用/agent/active/sessionid/接口，获取指定用户当前最近一次更新的会话ID，接受到前端用户请求:{user_id}")
    
    active_session_id = await app.state.session_manager.get_user_active_session_id(user_id)
    
    if not active_session_id:
        logger.error(f"用户 {user_id} 的会话不存在")
        return ActiveSessionInfoResponse(
            active_session_id=""
        )    
    response = ActiveSessionInfoResponse(
        active_session_id=active_session_id
    )
    logger.info(f"返回当前用户的最近一次更新的会话ID:{response}")
    return response

#API接口:获取指定用户的所有会话id
@app.get("/agent/sessionids/{user_id}", response_model=SessionInfoResponse)
async def get_agent_sessionids(user_id: str):
    logger.info(f"调用/agent/sessionids/接口，获取指定用户的所有会话ID，接受到前端用户请求:{user_id}")
    
    session_ids = await app.state.session_manager.get_all_session_ids(user_id)
    
    if not session_ids:
        logger.error(f"用户 {user_id} 的会话不存在")
        return SessionInfoResponse(
            session_ids=[]
        )    
    response = SessionInfoResponse(
        session_ids=session_ids
    )
    logger.info(f"返回当前用户的所有会话ID:{response}")
    return response
    

# API接口:删除指定用户当前会话
@app.delete("/agent/session/{user_id}/{session_id}")
async def delete_agent_session(user_id: str, session_id: str):
    logger.info(f"调用/agent/session/接口，删除指定用户当前会话，接受到前端用户请求:{user_id}:{session_id}")
    
    exists = await app.state.session_manager.session_id_exists(user_id, session_id)
    
    if not exists:
        logger.error(f"status_code=404,用户会话 {user_id}:{session_id} 不存在")
        raise HTTPException(status_code=404, detail=f"用户会话 {user_id}:{session_id} 不存在")
    
    #若会话存在，删除会话
    await app.state.session_manager.delete_session(user_id, session_id)
    
    logger.info(f"成功删除用户会话 {user_id}:{session_id}")
    return {"message": f"成功删除用户会话 {user_id}:{session_id}"}

# API接口:写入指定用户的长期记忆
@app.post("/agent/write/longterm")
async def write_long_term(request: LongMemRequest):
    logger.info(f"调用/agent/write/longterm接口，写入指定用户的长期记忆，接受到前端用户请求:{request}")
    
    user_id = request.user_id
    memory_info = request.memory_info
    
    exists = await app.state.session_manager.user_id_exists(user_id)
    # 如果不存在 则抛出异常
    if not exists:
        logger.error(f"status_code=404,用户 {user_id} 不存在")
        raise HTTPException(status_code=404, detail=f"用户会话 {user_id} 不存在")

    result = await write_long_term_info(user_id, memory_info)

        # 检查返回结果是否成功
    if result.get("success", False):
        # 构造成功响应
        return {
            "status": "success",
            "memory_id": result.get("memory_id"),
            "message": result.get("message", "记忆存储成功")
        }
    else:
        # 处理非成功返回结果
        raise HTTPException(
            status_code=500,
            detail="记忆存储失败，返回结果未包含成功状态"
        )
    

if __name__ == "__main__":
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)



   
