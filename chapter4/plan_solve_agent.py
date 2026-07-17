"""
4.3 Plan-and-Solve — 先计划后执行

与 ReAct 的区别:
  ReAct:    边走边看，每一步依赖上一轮的 Observation
  Plan-and-Solve: 先一次性生成完整计划，再逐步执行

架构:
  Planner  → 生成计划列表: ["步骤1", "步骤2", "步骤3"]
  Executor → 逐步执行每个步骤，上一步结果作为下一步上下文
"""

import ast
from .llm_client import HelloAgentsLLM

# ==================== Planner 提示词 ====================
PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划，```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# ==================== Executor 提示词 ====================
EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""


class Planner:
    """规划器：将复杂问题拆解为步骤列表"""

    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def plan(self, question: str) -> list[str]:
        """生成行动计划"""
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        print("\n--- 正在生成计划 ---")
        response_text = self.llm_client.think(messages=messages) or ""
        print(f"✅ 计划已生成:\n{response_text}")

        try:
            # 从 ```python ... ``` 代码块中提取列表
            plan_str = (
                response_text.split("```python")[1].split("```")[0].strip()
            )
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []

        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            return []


class Executor:
    """执行器：按计划逐步执行，每步结果作为下步上下文"""

    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def execute(self, question: str, plan: list[str]) -> str:
        """逐步执行计划，返回最后一步的结果"""
        history = ""
        print("\n--- 正在执行计划 ---")

        for i, step in enumerate(plan):
            print(f"\n-> 正在执行步骤 {i+1}/{len(plan)}: {step}")

            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan="\n".join([f"{j+1}. {s}" for j, s in enumerate(plan)]),
                history=history if history else "无",
                current_step=step,
            )
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages) or ""

            history += f"步骤 {i+1}: {step}\n结果: {response_text}\n\n"
            print(f"✅ 步骤 {i+1} 已完成，结果: {response_text}")

        return response_text  # 最后一步结果即为最终答案


class PlanAndSolveAgent:
    """
    Plan-and-Solve 智能体：先规划后执行。

    使用方式:
        llm = HelloAgentsLLM()
        agent = PlanAndSolveAgent(llm)
        agent.run("一个水果店周一卖了15个苹果，周二翻倍，周三比周二少5个，共卖多少？")
    """

    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.executor = Executor(self.llm_client)

    def run(self, question: str):
        print(f"\n{'='*50}")
        print(f"--- 开始处理问题 ---")
        print(f"问题: {question}")

        # 第 1 步：规划
        plan = self.planner.plan(question)
        if not plan:
            print("\n--- 任务终止 ---\n无法生成有效的行动计划。")
            return

        # 第 2 步：执行
        final_answer = self.executor.execute(question, plan)
        print(f"\n{'='*50}")
        print(f"--- 任务完成 ---")
        print(f"最终答案: {final_answer}")
        return final_answer
