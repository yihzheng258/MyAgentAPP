import os

class Config:
    
    # log config
    LOG_FILE = "logfile/app.log"
    if not os.path.exists(os.path.dirname(LOG_FILE)):
        os.makedirs(os.path.dirname(LOG_FILE))
    MAX_BYTES = 5*1024*1024,
    BACKUP_COUNT = 3
    
    # PostgreSQL config
    DB_URL = os.getenv("DB_URI", "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable")
    MIN_SIZE = 5
    MAX_SIZE = 10
    
    #redis config
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    SESSION_TIMEOUT = 300 # default 
    TTL = 3600 # 人为指定的key过期时间
    
    #LLM config
    LLM_TYPE = "openai"
    
    #API config
    HOST = "0.0.0.0"
    PORT = 8001