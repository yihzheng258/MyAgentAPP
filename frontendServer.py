import uuid
import requests
import json
import traceback
from typing import Dict, Any, Optional
import time
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown
from rich.theme import Theme
from rich.progress import Progress

custom_theme = Theme({
    "info": "cyan bold",
    "warning": "yellow bold",
    "success": "green bold",
    "error": "red bold",
    "heading": "magenta bold underline",
    "highlight": "blue bold",
})

console = Console(theme=custom_theme)

#后端API地址
API_BASE_URL = "http://localhost:8001"

def get_system_info():
    response = requests.get(f"{API_BASE_URL}/system/info")
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"获取系统信息失败: {response.status_code} - {response.text}")




def main():
    console.print(Panel(
        "前端客户端模拟服务",
        title="[heading]ReAct Agent智能体交互演示系统[/heading]",
        border_style="magenta"
    ))
    
    try:
        system_info = get_system_info()
        console.print(f"[info]当前系统内全部会话总计: {system_info['sessions_count']}[/info]")
        if system_info['active_users']:
            console.print(f"[info]系统内全部用户及用户会话: {system_info['active_users']}[/info]")
    except Exception:
        console.print("[warning]无法获取当前系统内会话状态信息[/warning]")
        
    #输入用户ID，没有就用默认值
    default_user_id = f"user_{int(time.time())}"
    user_id = Prompt.ask("[info]请输入用户ID[/info] (新ID将创建新用户，已有ID将恢复使用该用户)", default=default_user_id)

    

if __name__ == "__main__":
    main()
