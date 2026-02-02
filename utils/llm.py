import os
import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from .config import Config

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


MODEL_CONFIGS = {
    "openai": {
        "base_url": "https://nangeai.top/v1",
        "api_key": "sk-123",
        "chat_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small"
    }
}

# 默认配置
DEFAULT_LLM_TYPE = "openai"
DEFAULT_TEMPERATURE = 0

class LLMInitializationError(Exception):
    pass


