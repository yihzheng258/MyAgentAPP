import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
from typing import Callable
from langchain_core.tools import BaseTool, tool as create_tool
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.interrupt import HumanInterruptConfig, HumanInterrupt
from langgraph.types import interrupt, Command
from langchain_core.tools import tool
from .config import Config
from langchain_mcp_adapters.client import MultiServerMCPClient

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

# 添加human-in-the-loop功能到工具
async def add_human_in_the_loop(
    tool: Callable | BaseTool,
    *,
    interrupt_config: HumanInterruptConfig = None
) -> BaseTool:
    
    if not isinstance(tool, BaseTool):
        tool = create_tool(tool)
        
    @create_tool(
        tool.name,
        description=tool.description,
        args_schema=tool.args_schema
    )
    async def call_tool_with_interrupt(config: RunnableConfig, **tool_input):
        
        # 创建中断
        request: HumanInterrupt = {
            "action_request": {
                "action": tool.name,
                "args": tool_input
            },
            "config": interrupt_config,
            "description": f"准备调用 {tool.name} 工具：\n- 参数为: {tool_input}\n\n是否允许继续？\n输入 'yes' 接受工具调用\n输入 'no' 拒绝工具调用\n输入 'edit' 修改工具参数后调用工具\n输入 'response' 不调用工具直接反馈信息",
        }
        
        #利用interrupt机制进行人工审查
        response = interrupt(request)
        logger.info(f"response: {response}")
        
        if response["type"] == "accept":
            logger.info("工具调用已批准，执行中...")
            logger.info(f"调用工具: {tool.name}, 参数: {tool_input}")
            try:
                tool_response = await tool.ainvoke(input=tool_input)
                logger.info(tool_response)
            except Exception as e:
                logger.error(f"工具调用失败: {e}")
        
        elif response["type"] == "edit":
            tool_input = response["args"]["args"]
            try:
                # 使用更新后的参数调用原始工具
                tool_response = await tool.ainvoke(input=tool_input)
                logger.info(tool_response)
            except Exception as e:
                logger.error(f"工具调用失败: {e}")
        
        elif response["type"] == "reject":
            logger.info("工具调用被拒绝，等待用户输入...")
            # 直接将用户反馈作为工具的响应
            tool_response = '该工具被拒绝使用，请尝试其他方法或拒绝回答问题。'
        
        elif response["type"] == "response":
            # 如果是响应，直接将用户反馈作为工具的响应
            user_feedback = response["args"]
            tool_response = user_feedback

        else:
            raise ValueError(f"Unsupported interrupt response type: {response['type']}")

        return tool_response
    
    return call_tool_with_interrupt

# 提供高德MCP tools
async def get_tools():
    
    # MCP Server工具 高德地图
    client = MultiServerMCPClient({
        # 高德地图MCP Server
        "amap-amap-sse": {
            "url": "https://mcp.amap.com/sse?key=848232bewe1987634de9ew23e19wewed61265e50bb0757",
            "transport": "sse",
        }
    })
    
    # 从MCP Server中获取可提供使用的全部工具
    amap_tools = await client.get_tools()
    # 为工具添加人工审查
    tools = [await add_human_in_the_loop(index) for index in amap_tools]

    return tools