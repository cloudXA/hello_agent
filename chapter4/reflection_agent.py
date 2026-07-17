"""
4.4 Reflection — 自我反思与迭代优化

Reflection 范式赋予 Agent "自我批判"能力：
  初始执行 → 评审反思 → 根据反馈优化 → 再反思 → ... → 收敛

典型场景：代码生成、写作、翻译等需要反复打磨的任务。

架构:
  Memory      → 存储每一轮的"执行轨迹"和"评审反馈"
  Agent Loop  → 执行 → 反思 → 优化 → 下一轮（直到"无需改进"或达到最大轮数）
"""

from typing import List, Dict, Optional
from .llm_client import HelloAgentsLLM

# ==================== Memory 模块 ====================
class Memory:
    """
    短期记忆：存储执行轨迹与反思反馈。

    记录类型:
      - "execution":  模型生成的代码/内容
      - "reflection": 评审员的反馈意见
    """

    def __init__(self):
        self.records: List[Dict[str, any]] = []

    def add_record(self, record_type: str, content: str):
        self.records.append({"type": record_type, "content": content})
        print(f"📝 记忆已更新，新增一条 '{record_type}' 记录。")

    def get_trajectory(self) -> str:
        """返回完整轨迹（代码 + 反馈），供模型参考"""
        parts = []
        for record in self.records:
            if record["type"] == "execution":
                parts.append(f"--- 上一轮尝试 (代码) ---\n{record['content']}")
            elif record["type"] == "reflection":
                parts.append(f"--- 评审员反馈 ---\n{record['content']}")
        return "\n\n".join(parts)

    def get_last_execution(self) -> Optional[str]:
        """获取最近一次的执行结果"""
        for record in reversed(self.records):
            if record["type"] == "execution":
                return record["content"]
        return None


# ==================== 提示词模板 ====================

INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求: {task}

请直接输出代码，不要包含任何额外的解释。
"""

REFLECT_PROMPT_TEMPLATE = """
你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下Python代码，并专注于找出其在**算法效率**上的主要瓶颈。

# 原始任务:
{task}

# 待审查的代码:
```python
{code}
```

请分析该代码的时间复杂度，并思考是否存在一种**算法上更优**的解决方案来显著提升性能。
如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法替代试除法）。
如果代码在算法层面已经达到最优，才能回答"无需改进"。

请直接输出你的反馈，不要包含任何额外的解释。
"""

REFINE_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务:
{task}

# 你上一轮尝试的代码:
{last_code_attempt}

评审员的反馈：
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
"""


class ReflectionAgent:
    """
    Reflection 智能体：通过自我反思迭代优化输出。

    适用场景：代码生成、文案优化、翻译润色等需要反复打磨的任务。

    使用方式:
        llm = HelloAgentsLLM()
        agent = ReflectionAgent(llm, max_iterations=3)
        agent.run("编写一个Python函数，找出1到n之间所有的素数")
    """

    def __init__(self, llm_client: HelloAgentsLLM, max_iterations: int = 3):
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iterations = max_iterations

    def _get_llm_response(self, prompt: str) -> str:
        """调用 LLM 获取响应"""
        messages = [{"role": "user", "content": prompt}]
        return self.llm_client.think(messages=messages) or ""

    def run(self, task: str):
        print(f"\n{'='*50}")
        print(f"--- 开始处理任务 ---")
        print(f"任务: {task}")

        # ========== 第 1 阶段：初始执行 ==========
        print("\n--- 正在进行初始尝试 ---")
        initial_code = self._get_llm_response(
            INITIAL_PROMPT_TEMPLATE.format(task=task)
        )
        self.memory.add_record("execution", initial_code)

        # ========== 第 2 阶段：迭代反思与优化 ==========
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1}/{self.max_iterations} 轮迭代 ---")

            # a. 反思
            print("\n-> 正在进行反思...")
            last_code = self.memory.get_last_execution()
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(
                task=task, code=last_code
            )
            feedback = self._get_llm_response(reflect_prompt)
            self.memory.add_record("reflection", feedback)

            # b. 终止条件：评审员认为"无需改进"
            if "无需改进" in feedback:
                print("\n✅ 反思认为代码已无需改进，任务完成。")
                break

            # c. 根据反馈优化
            print("\n-> 正在进行优化...")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task,
                last_code_attempt=last_code,
                feedback=feedback,
            )
            refined_code = self._get_llm_response(refine_prompt)
            self.memory.add_record("execution", refined_code)

        # ========== 输出最终结果 ==========
        final_code = self.memory.get_last_execution()
        print(f"\n{'='*50}")
        print(f"--- 任务完成 ---")
        print(f"最终生成的代码:")
        print(final_code if final_code else "未能生成有效代码")

        # 打印完整轨迹供审计
        print(f"\n--- 完整执行轨迹 ---")
        print(self.memory.get_trajectory())

        return final_code
