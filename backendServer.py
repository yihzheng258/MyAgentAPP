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
import uvicorn
from contextlib import asynccontextmanager
import redis.asyncio as redis
import json
from datetime import timedelta
from psycopg_pool import AsyncConnectionPool
from utils.config import Config
from utils.llms import get_llm
from utils.tools import get_tools


# 设置日志基本配置，级别为DEBUG或INFO
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.handlers = []  
handler = ConcurrentRotatingFileHandler(
    Config.LOG_FILE,
    maxBytes = Config.MAX_BYTES,
    backupCount = Config.BACKUP_COUNT
)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)


