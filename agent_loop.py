"""
第一章 1.3.3 执行行动循环
- 组装 prompt、调用 LLM、正则解析 Thought/Action
- 执行工具调用、收集 Observation、驱动智能体完成任务的 main loop

========================================
本文档中每一处正则都配有实例讲解，方便理解。
模型输出格式是由 AGENT_SYSTEM_PROMPT 约定的：
  Thought: [思考内容]
  Action: function_name(arg="value")
  或
  Action: Finish[最终答案]
"""

import re
import os
from dotenv import load_dotenv

# 导入前面写的模块
from llm_client import OpenAICompatibleClient
from chapter1_prepare import AGENT_SYSTEM_PROMPT, available_tools

load_dotenv()


def run_agent(user_prompt: str, max_iterations: int = 5):
    """执行智能体主循环，最多 max_iterations 轮。"""
    # ============ 1. 配置 LLM 客户端 ============
    API_KEY = os.getenv("OPENAI_API_KEY", "")
    BASE_URL = os.getenv("OPENAI_BASE_URL", "")
    MODEL_ID = os.getenv("OPENAI_MODEL", "")

    llm = OpenAICompatibleClient(
        model=MODEL_ID,
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # ============ 2. 初始化 ============
    prompt_history = [f"用户请求: {user_prompt}"]
    print(f"用户输入: {user_prompt}\n" + "=" * 40)

    # ============ 3. 主循环 ============
    for i in range(max_iterations):
        print(f"--- 循环 {i + 1} ---\n")

        # 3.1 构建 Prompt
        full_prompt = "\n".join(prompt_history)

        # ============================================================
        # 3.2 调用 LLM
        # ============================================================
        #
        # 模型原始输出有两种可能：
        #   【A — 调用工具】  Thought: 查天气\nAction: get_weather(city="北京")
        #   【B — 结束任务】  Thought: 信息够了\nAction: Finish[北京晴天，去故宫]
        #
        # 但模型有时"多嘴"，一次吐出多对 Thought-Action：
        #   Thought: 查天气
        #   Action: get_weather(city="北京")
        #
        #   Thought: 查到了，搜景点     ← 多余的，必须砍掉
        #   Action: get_attraction(...) ← 这一轮还没拿到天气结果，不该提前想下一步
        #
        # 所以下面的正则①用来截断，只取第一对。

        print(f"【发给模型的 Prompt】:\n{full_prompt}\n" + "-" * 40)

        llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
        print(f"【模型原始输出】:\n{llm_output}\n" + "-" * 40)

        # ---- 正则①：截断多余的 Thought-Action 对，只保留第一对 ----
        # r'(Thought:.*?Action:.*?)'            捕获组：从 "Thought:" 到第一个 "Action:" 结束
        #   .*? = 非贪婪匹配，尽量少匹配，确保停在第一个 Action: 之后
        #
        # r'(?=...)'                            前瞻断言：必须紧跟以下内容，但不消耗字符
        #   \n\s*(?:Thought:|Action:|Observation:)  下一行出现了第二个 Thought/Action/Observation
        #   |\Z                                     或者已经是字符串末尾
        #
        # 实例：
        #   输入 → "Thought: A\nAction: X\n\nThought: B\nAction: Y"
        #   匹配 → "Thought: A\nAction: X"          ← .group(1) 只拿第一对
        #
        #   (?= 命中了 "\n\nThought: B" 中的 "\n\nThought:") → 所以停在 X 后面
        match = re.search(
            r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)',
            llm_output, re.DOTALL
        )
        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                llm_output = truncated
                print("⚠️ 已截断多余的 Thought-Action 对")

        print(f"【处理后的输出】:\n{llm_output}\n")
        prompt_history.append(llm_output)

        # ============================================================
        # 3.3 提取 Action 内容
        # ============================================================
        #
        # ---- 正则②：提取 Action: 后面的所有内容 ----
        # r"Action: (.*)"   匹配 "Action: " 字面量，然后捕获后面所有字符
        #
        # 实例：
        #   输入 → "Thought: 查天气\nAction: get_weather(city="北京")"
        #   匹配 → "get_weather(city="北京")"   ← .group(1)
        action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
        if not action_match:
            observation = "错误: 未能解析到 Action 字段。"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "=" * 40)
            prompt_history.append(observation_str)
            continue

        action_str = action_match.group(1).strip()

        # ============================================================
        # 3.4 判断：调用工具 vs 结束任务
        # ============================================================
        #
        # ---- 正则③：提取 Finish[最终答案] 中的内容 ----
        # r"Finish\[(.*)\]"    匹配 "Finish[" 字面量，捕获 [] 内的所有内容
        #   注意：[] 在正则中是元字符，所以要用 \[ 和 \] 转义
        #
        # 实例：
        #   输入 → "Finish[北京晴天25度，推荐去故宫。]"
        #   匹配 → "北京晴天25度，推荐去故宫。"   ← .group(1)
        if action_str.startswith("Finish"):
            finish_match = re.match(r"Finish\[(.*)\]", action_str)
            final_answer = finish_match.group(1) if finish_match else action_str
            print(f"✅ 模型决定结束任务")
            print(f"最终答案: {final_answer}")
            return final_answer

        # ============================================================
        # 3.5 不是 Finish → 解析工具名和参数
        # ============================================================
        #
        # action_str 示例：get_weather(city="北京")
        #
        # ---- 正则④：提取工具名 ----
        # r"(\w+)\("       捕获括号前的连续字母/数字/下划线
        #
        # 实例：
        #   输入 → "get_weather(city="北京")"
        #   匹配 → "get_weather"   ← .group(1)
        tool_match = re.search(r"(\w+)\(", action_str)
        if not tool_match:
            observation = f"错误: 无法解析工具调用 - {action_str}"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "=" * 40)
            prompt_history.append(observation_str)
            continue

        tool_name = tool_match.group(1)

        # ---- 正则⑤：提取括号内的参数字符串 ----
        # r"\((.*)\)"      匹配括号内的所有内容
        #
        # 实例：
        #   输入 → "get_weather(city="北京")"
        #   匹配 → 'city="北京"'   ← .group(1)
        args_str = re.search(r"\((.*)\)", action_str).group(1)

        # ---- 正则⑥：将 key="value" 对解析成字典 ----
        # r'(\w+)="([^"]*)"'  findall 返回所有匹配的 (key, value) 元组列表
        #   (\w+)    捕获参数名（字母/数字/下划线）
        #   ="      字面量 ="
        #   ([^"]*)  捕获参数值（非引号的任意字符）
        #   "        字面量结尾引号
        #
        # 实例：
        #   输入 → 'city="北京", weather="晴"'
        #   findall → [('city', '北京'), ('weather', '晴')]
        #   dict()  → {"city": "北京", "weather": "晴"}
        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

        # ---- 3.6 调用工具 ----
        # available_tools[tool_name](**kwargs)
        #   等价于 get_weather(city="北京")
        #
        # ** 是 Python 字典解包操作符：
        #   func(**{"city": "北京"})  ≡  func(city="北京")
        if tool_name in available_tools:
            observation = available_tools[tool_name](**kwargs)
        else:
            observation = f"错误:未定义的工具 '{tool_name}'"

        # ---- 3.7 记录 Observation ----
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "=" * 40)
        prompt_history.append(observation_str)

    return "错误: 达到最大循环次数，智能体未能完成任务。"


# ===================== 入口 =====================
if __name__ == "__main__":
    result = run_agent("你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。")
    print("\n" + "=" * 40)
    print(f"最终结果: {result}")
