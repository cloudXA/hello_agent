"""
第一章 1.3.3 执行行动循环
- 组装 prompt、调用 LLM、正则解析 Thought/Action
- 执行工具调用、收集 Observation、驱动智能体完成任务的 main loop
"""

import re
import os
from dotenv import load_dotenv

# 导入前面写的模块
from llm_client import OpenAICompatibleClient
from chapter1_prepare import AGENT_SYSTEM_PROMPT, available_tools

# 加载 .env 环境变量
load_dotenv()


def run_agent(user_prompt: str, max_iterations: int = 5):
    """执行智能体主循环，最多 max_iterations 轮。"""
    # --- 1. 配置 LLM 客户端 ---
    API_KEY = os.getenv("OPENAI_API_KEY", "")
    BASE_URL = os.getenv("OPENAI_BASE_URL", "")
    MODEL_ID = os.getenv("OPENAI_MODEL", "")

    llm = OpenAICompatibleClient(
        model=MODEL_ID,
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # --- 2. 初始化 ---
    prompt_history = [f"用户请求: {user_prompt}"]
    print(f"用户输入: {user_prompt}\n" + "=" * 40)

    # --- 3. 运行主循环 ---
    for i in range(max_iterations):
        print(f"--- 循环 {i + 1} ---\n")

        # 3.1. 构建 Prompt
        full_prompt = "\n".join(prompt_history)

        # 3.2. 调用LLM进行思考
        llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
        # 模型可能会输出多余的Thought-Action，需要截断
        match = re.search(
            r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)',
            llm_output, re.DOTALL
        )
        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                llm_output = truncated
                print("已截断多余的 Thought-Action 对")
        print(f"模型输出:\n{llm_output}\n")
        prompt_history.append(llm_output)

        # 3.3. 解析并执行行动
        action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
        if not action_match:
            observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "=" * 40)
            prompt_history.append(observation_str)
            continue

        action_str = action_match.group(1).strip()

        # 3.4. 检查是否是结束指令
        if action_str.startswith("Finish"):
            finish_match = re.match(r"Finish\[(.*)\]", action_str)
            final_answer = finish_match.group(1) if finish_match else action_str
            print(f"任务完成，最终答案: {final_answer}")
            return final_answer

        # 3.5. 解析工具名和参数并执行
        tool_match = re.search(r"(\w+)\(", action_str)
        if not tool_match:
            observation = f"错误: 无法解析工具调用 - {action_str}"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "=" * 40)
            prompt_history.append(observation_str)
            continue

        tool_name = tool_match.group(1)
        args_str = re.search(r"\((.*)\)", action_str).group(1)
        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

        if tool_name in available_tools:
            observation = available_tools[tool_name](**kwargs)
        else:
            observation = f"错误:未定义的工具 '{tool_name}'"

        # 3.6. 记录观察结果
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "=" * 40)
        prompt_history.append(observation_str)

    # 循环结束仍未 Finish，返回错误
    return "错误: 达到最大循环次数，智能体未能完成任务。"


# ===================== 入口 =====================
if __name__ == "__main__":
    # 运行智能体：查询北京天气并推荐景点
    result = run_agent("你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。")
    print("\n" + "=" * 40)
    print(f"最终结果: {result}")
