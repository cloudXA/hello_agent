"""
第一章 1.3.2 接入大语言模型
- OpenAICompatibleClient: 调用任何兼容 OpenAI 接口的 LLM 服务的通用客户端
- 配置从 .env 读取 (OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL)
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# 从项目根目录的 .env 文件加载环境变量
load_dotenv()


class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端。
    """
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用LLM API来生成回应。"""
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"


# ===================== 配置 (从 .env 读取) =====================
# 这三个值取决于你使用的服务商（OpenAI 官方 / Azure / Ollama / vLLM 等）
API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "")
MODEL_ID = os.getenv("OPENAI_MODEL", "")



