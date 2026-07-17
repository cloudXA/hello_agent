"""
4.2 ReAct 模式的工具定义
- search: 基于 SerpApi 的网页搜索工具
- ToolExecutor: 工具注册与执行器
"""

import os
from typing import Dict, Callable, Any


def search(query: str) -> str:
    """
    基于 SerpApi 的网页搜索工具。优先返回答案框或知识图谱内容。

    需要配置 SERPAPI_API_KEY 环境变量（去 https://serpapi.com/ 注册获取免费额度）。
    如果没有配置，会返回模拟结果让演示继续运行。
    """
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")

    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        # 没有配置 API Key 时的模拟返回，方便本地演示
        return (
            f"[模拟搜索结果] 关于 '{query}' 的搜索:\n"
            f"这是一条模拟的搜索结果。\n"
            f"要获得真实搜索能力，请在 .env 中配置 SERPAPI_API_KEY。\n"
            f"注册地址: https://serpapi.com/"
        )

    try:
        from serpapi import SerpApiClient

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",
            "hl": "zh-cn",
        }
        client = SerpApiClient(params)
        results = client.get_dict()

        # 智能解析：按优先级返回最有价值的信息
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

        return f"对不起，没有找到关于 '{query}' 的信息。"

    except ImportError:
        return (
            "错误: 未安装 serpapi 库。请运行: pip install google-search-results"
        )
    except Exception as e:
        return f"搜索时发生错误: {e}"


class ToolExecutor:
    """一个工具执行器，负责管理和执行工具。"""

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: Callable):
        """向工具箱中注册一个新工具。"""
        if name in self.tools:
            print(f"警告: 工具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> Callable:
        """根据名称获取一个工具的执行函数。"""
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """获取所有可用工具的格式化描述字符串。"""
        return "\n".join(
            [f"- {name}: {info['description']}" for name, info in self.tools.items()]
        )


if __name__ == "__main__":
    # 测试工具注册与调用
    executor = ToolExecutor()
    executor.registerTool("Search", "一个网页搜索引擎，用于查询实时或未知信息。", search)
    print("可用工具:\n" + executor.getAvailableTools())
    print("\n搜索测试:")
    print(search("北京今天天气"))
