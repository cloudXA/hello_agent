"""
4.1.3 封装基础 LLM 调用函数
- HelloAgentsLLM: 兼容 OpenAI 接口的 LLM 客户端，默认流式输出
- 优先使用 LLM_* 环境变量，回退到 OPENAI_* 环境变量（兼容第一章配置）
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()


class HelloAgentsLLM:
    """
    为 Hello Agents 定制的 LLM 客户端。
    调用任何兼容 OpenAI 接口的服务，默认使用流式响应。
    """

    def __init__(
        self,
        model: str = None,
        apiKey: str = None,
        baseUrl: str = None,
        timeout: int = None,
    ):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        兼容两套命名：LLM_* (第四章) 和 OPENAI_* (第一章)
        """
        self.model = (
            model
            or os.getenv("LLM_MODEL_ID")
            or os.getenv("OPENAI_MODEL")
        )
        apiKey = (
            apiKey
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        baseUrl = (
            baseUrl
            or os.getenv("LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
        )
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError(
                "模型ID、API密钥和服务地址必须被提供或在.env文件中定义。\n"
                "请在 .env 中配置 LLM_MODEL_ID / LLM_API_KEY / LLM_BASE_URL\n"
                "或 OPENAI_MODEL / OPENAI_API_KEY / OPENAI_BASE_URL"
            )

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其流式响应拼接后的完整文本。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None


# --- 客户端使用示例 ---
if __name__ == "__main__":
    try:
        llmClient = HelloAgentsLLM()

        exampleMessages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that writes Python code.",
            },
            {"role": "user", "content": "写一个快速排序算法"},
        ]

        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)
