"""
4.2 ReAct 模式的工具定义
- search: 网页搜索工具（Tavily > SerpApi > 模拟兜底）
- ToolExecutor: 工具注册与执行器
"""

import os
from typing import Dict, Callable, Any
from dotenv import load_dotenv

load_dotenv()


def search(query: str) -> str:
    """
    网页搜索工具。按优先级自动选择后端：
      1. Tavily（如果你配置了 TAVILY_API_KEY）
      2. SerpApi（如果你配置了 SERPAPI_API_KEY）
      3. 模拟数据（都没配置时，保证演示不中断）

    去 https://www.tavily.com/ 注册获取免费额度。
    """
    print(f"🔍 正在执行 [搜索] 网页搜索: {query}")

    # ---- 后端 1：Tavily ----
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=tavily_key)
            result = client.search(query=query, search_depth="basic")
            if result.get("answer"):
                return result["answer"]
            snippets = [
                f"[{i+1}] {r.get('title', '')}\n{r.get('content', '')}"
                for i, r in enumerate(result.get("results", [])[:3])
            ]
            if snippets:
                return "\n\n".join(snippets)
            return f"未找到关于 '{query}' 的相关信息。"
        except Exception as e:
            print(f"Tavily 搜索失败: {e}，尝试下一后端...")

    # ---- 后端 2：SerpApi ----
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if serpapi_key:
        try:
            from serpapi import SerpApiClient

            params = {
                "engine": "google",
                "q": query,
                "api_key": serpapi_key,
                "gl": "cn",
                "hl": "zh-cn",
            }
            client = SerpApiClient(params)
            results = client.get_dict()

            if "answer_box_list" in results:
                return "\n".join(results["answer_box_list"])
            if "answer_box" in results and "answer" in results["answer_box"]:
                return results["answer_box"]["answer"]
            if (
                "knowledge_graph" in results
                and "description" in results["knowledge_graph"]
            ):
                return results["knowledge_graph"]["description"]
            if "organic_results" in results and results["organic_results"]:
                snippets = [
                    f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                    for i, res in enumerate(results["organic_results"][:3])
                ]
                return "\n\n".join(snippets)
            return f"未找到关于 '{query}' 的信息。"
        except ImportError:
            print("SerpApi 未安装，跳过...")
        except Exception as e:
            print(f"SerpApi 搜索失败: {e}")

    # ---- 兜底：模拟数据 ----
    return (
        f"[模拟搜索结果] 关于 '{query}' 的搜索:\n"
        f"这是一条模拟的搜索结果。\n"
        f"要获得真实搜索能力，请在 .env 中配置 TAVILY_API_KEY（推荐）或 SERPAPI_API_KEY。\n"
        f"Tavily 注册地址: https://www.tavily.com/"
    )


class ToolExecutor:
    """一个工具执行器，负责管理和执行工具。"""

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: Dict[str, Any] = None,
    ):
        """
        向工具箱中注册一个新工具。

        参数:
            name: 工具名称
            description: 工具描述
            func: 工具执行函数
            parameters: JSON Schema 格式的参数定义（用于 function calling）
        """
        if name in self.tools:
            print(f"警告: 工具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {
            "description": description,
            "func": func,
            "parameters": parameters or {},
        }
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> Callable:
        """根据名称获取一个工具的执行函数。"""
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """获取所有可用工具的格式化描述字符串（正则解析版用）。"""
        return "\n".join(
            [f"- {name}: {info['description']}" for name, info in self.tools.items()]
        )

    def to_openai_tools(self) -> list[Dict[str, Any]]:
        """
        将已注册的工具转换为 OpenAI function calling 格式。

        返回:
            [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "网页搜索引擎...",
                        "parameters": { ... JSON Schema ... }
                    }
                },
                ...
            ]
        """
        tool_specs = []
        for name, info in self.tools.items():
            spec = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": info.get("parameters", {}),
                },
            }
            # 如果没提供 parameters，生成一个默认的 query 参数 schema
            if not spec["function"]["parameters"]:
                spec["function"]["parameters"] = {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"传递给 {name} 工具的输入参数",
                        }
                    },
                    "required": ["query"],
                }
            tool_specs.append(spec)
        return tool_specs


if __name__ == "__main__":
    # 测试工具注册与调用
    executor = ToolExecutor()
    executor.registerTool("Search", "一个网页搜索引擎，用于查询实时或未知信息。", search)
    print("可用工具:\n" + executor.getAvailableTools())
    print("\n搜索测试:")
    print(search("北京今天天气"))
