"""
4.2 ReAct (Reasoning and Acting) — 边想边做

ReAct 是 Agent 最经典的范式：
  Thought（思考） → Action（行动） → Observation（观察） → 下一轮

两种实现:
  - ReActAgent（正则解析版）: 用 prompt + 正则提取 Thought/Action
  - ReActAgentV2（Function Calling 版）: 用 OpenAI 原生 tools API，模型返回结构化 JSON

和第一章 agent_loop.py 的区别：
  - 使用 HelloAgentsLLM（流式输出 + .env 自动配置）
  - 使用 ToolExecutor（工具注册/管理更规范）
  - 提示词模板化（REACT_PROMPT_TEMPLATE）
"""

import re
import json
from .llm_client import HelloAgentsLLM
from .tools import ToolExecutor

# ==================== 提示词模板 ====================
REACT_PROMPT_TEMPLATE = """
你是一个有能力调用外部工具的智能助手。

## 可用工具
{tools}

## 交互规则（极其重要）
你只能输出 **一轮** 思考和行动，然后等待外部系统返回观察结果。
**绝对禁止**自己编造 Observation 或连续输出多轮 Action。
外部系统会把真实的观察结果返回给你，你再据此决定下一步。

## 输出格式（严格遵循，只输出一次）
Thought: 你的思考过程，用于分析当前情况、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`: 调用一个可用工具（tool_input 仅写搜索关键词，不要附带其他内容）。
- `Finish[最终答案]`: 当你已经获得足够信息可以回答用户时。

## 当前问题
Question: {question}

## 历史记录
{history}

现在请输出本轮你的 Thought 和 Action（只输出一轮，等待 Observation）:
"""


class ReActAgent:
    """
    ReAct 智能体：将"思考"与"行动"紧密结合。
    每一步都先思考再行动，根据观察结果动态调整下一步。

    使用方式:
        llm = HelloAgentsLLM()
        executor = ToolExecutor()
        executor.registerTool("Search", "网页搜索", search)
        agent = ReActAgent(llm, executor)
        agent.run("华为最新手机是哪一款？")
    """

    def __init__(
        self,
        llm_client: HelloAgentsLLM,
        tool_executor: ToolExecutor,
        max_steps: int = 5,
    ):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def _parse_output(self, text: str):
        """
        从模型输出中提取 Thought 和 Action。
        只取第一个 Action，防止 LLM 在单次回复中自导自演多轮循环。

        输入:  "Thought: 需要搜索\nAction: Search[华为最新手机]"
        返回:  ("需要搜索", "Search[华为最新手机]")
        """
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL
        )
        # 只取第一个 Action，以换行或 Observation 或第二个 Action 为边界截断
        action_match = re.search(
            r"Action:\s*(.*?)(?=\n(?:Observation|Action|Thought):|$)",
            text, re.DOTALL
        )

        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """
        解析 Action 字符串，返回 (工具名, 输入参数)。
        使用非贪婪匹配，只取第一个方括号对。

        输入:  "Search[华为最新手机]"
        返回:  ("Search", "华为最新手机")

        输入:  "Finish[华为P70 Pro是华为最新手机]"
        返回:  ("Finish", ...)
        """
        match = re.match(r"(\w+)\[(.*?)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def run(self, question: str):
        """运行 ReAct 循环"""
        self.history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n{'='*50}")
            print(f"--- 第 {current_step} 步 ---")

            # 1. 构建 Prompt
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc, question=question, history=history_str
            )
            messages = [{"role": "user", "content": prompt}]

            # 2. 调用 LLM
            response_text = self.llm_client.think(messages=messages)
            if not response_text:
                print("错误: LLM未能返回有效响应。")
                break

            # 3. 解析 Thought / Action
            thought, action = self._parse_output(response_text)
            if thought:
                print(f"💭 思考: {thought}")
            if not action:
                print("警告: 未能解析出有效的 Action，流程终止。")
                break

            # 4. 判断：调用工具 还是 结束任务？
            if action.startswith("Finish"):
                finish_match = re.match(r"Finish\[(.*)\]", action, re.DOTALL)
                final_answer = (
                    finish_match.group(1) if finish_match else action
                )
                print(f"🎉 最终答案: {final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or tool_input is None:
                print(f"警告: 无效的 Action 格式 -> {action}")
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")

            # 5. 执行工具
            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误: 未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_function(tool_input)

            print(f"👀 观察: {observation}")

            # 6. 更新历史
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        print("已达到最大步数，流程终止。")
        return None


# ==================== V2: Function Calling 版 ====================

SYSTEM_PROMPT_V2 = """你是一个有能力调用外部工具的智能助手。

## 工作方式
你可以使用系统提供的工具来获取信息，帮助用户解答问题。
每一步调用工具后，系统会将真实的观察结果返回给你，你再据此决定下一步。

## 重要规则
- 如果工具返回的信息不足以回答问题，可以继续调用其他工具
- 信息充分时，直接给出最终答案
- 不确定的信息要明确标注"不确定"，不编造不存在的结论
"""


class ReActAgentV2:
    """
    ReAct V2：使用 OpenAI 原生 function calling，零正则解析。

    与 ReActAgent（正则版）的对比:

    ┌──────────────────┬─────────────────────┬───────────────────────┐
    │ 维度             │ ReActAgent (正则)    │ ReActAgentV2 (FC)     │
    ├──────────────────┼─────────────────────┼───────────────────────┤
    │ 工具传递方式      │ Prompt 文本拼接       │ API tools 参数         │
    │ 输出格式          │ 自然语言 + 正则提取   │ 结构化 JSON            │
    │ 参数约束          │ 无（靠 prompt 约定）   │ JSON Schema 约束       │
    │ 解析失败概率      │ 中等（格式漂移）       │ 极低（API 保证）        │
    │ 多工具并行调用    │ 不支持               │ 原生支持               │
    │ 模型要求          │ 任何 LLM             │ 支持 function calling  │
    └──────────────────┴─────────────────────┴───────────────────────┘

    使用方式:
        llm = HelloAgentsLLM()
        executor = ToolExecutor()
        executor.registerTool("Search", "网页搜索", search)
        agent = ReActAgentV2(llm, executor)
        agent.run("华为最新手机是哪一款？")
    """

    def __init__(
        self,
        llm_client: HelloAgentsLLM,
        tool_executor: ToolExecutor,
        max_steps: int = 5,
    ):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps

    def run(self, question: str):
        """运行 function calling 驱动的 ReAct 循环"""
        # 构建 OpenAI tools 格式
        tools = self.tool_executor.to_openai_tools()

        # 初始化消息列表：system prompt + 用户问题
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_V2},
            {"role": "user", "content": question},
        ]

        print(f"\n{'='*50}")
        print(f"问题: {question}")
        print(f"可用工具: {[t['function']['name'] for t in tools]}")

        for step in range(1, self.max_steps + 1):
            print(f"\n{'='*50}")
            print(f"--- 第 {step} 步 ---")

            # 1. 调用 LLM（带 tools 定义）
            response_msg = self.llm_client.think_with_tools(
                messages=messages, tools=tools
            )
            if not response_msg:
                print("错误: LLM 未能返回有效响应。")
                break

            # 2. 如果模型直接返回文本（无 tool_calls），就是最终答案
            if not response_msg.tool_calls:
                final_answer = response_msg.content or ""
                print(f"🎉 最终答案: {final_answer}")
                return final_answer

            # 3. 处理 tool_calls：执行工具并收集结果
            # 将模型的 assistant 消息加入历史
            messages.append(response_msg)

            for tool_call in response_msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # 对于单参数 search(query)，取第一个参数值
                tool_input = (
                    tool_args.get("query", "")
                    or next(iter(tool_args.values()), "")
                )

                print(f"💭 思考: 需要调用 {tool_name} 工具")
                print(f"🎬 行动: {tool_name}({tool_args})")

                # 4. 执行工具
                tool_function = self.tool_executor.getTool(tool_name)
                if not tool_function:
                    observation = f"错误: 未找到名为 '{tool_name}' 的工具。"
                else:
                    observation = tool_function(tool_input)

                print(f"👀 观察: {observation[:200]}...")

                # 5. 将工具结果作为 tool 消息追加
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": observation,
                })

        print("已达到最大步数，流程终止。")
        return None
