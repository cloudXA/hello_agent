"""
第四章 智能体经典范式 — 演示入口

运行方式:
    cd Hello-Agents
    python -m chapter4.main [react|react2|plan|reflection|compare|all]

三大范式对比:

    ┌──────────┬──────────────────┬─────────────────────┬──────────────────┐
    │ 范式     │ 核心思想          │ 适用场景             │ 决策方式          │
    ├──────────┼──────────────────┼─────────────────────┼──────────────────┤
    │ ReAct    │ 边走边看          │ 需要工具调用的任务    │ 每步动态决定      │
    │ Plan-Solve│ 先规划后执行      │ 多步推理的复杂问题    │ 先全盘规划        │
    │ Reflection│ 反思 + 迭代优化   │ 代码/文案/翻译润色   │ 自我批判修正      │
    └──────────┴──────────────────┴─────────────────────┴──────────────────┘
"""

from chapter4.llm_client import HelloAgentsLLM
from chapter4.tools import search, ToolExecutor
from chapter4.react_agent import ReActAgent, ReActAgentV2
from chapter4.plan_solve_agent import PlanAndSolveAgent
from chapter4.reflection_agent import ReflectionAgent

# 测试用问题
TEST_QUESTION = (
    "2026年7月上映了哪些值得关注的电影？"
)


def demo_react():
    """演示 ReAct（正则解析版）：Prompt 拼接工具 + 正则提取 Thought/Action"""
    print("\n" + "█" * 50)
    print("█  演示 1: ReAct — 正则解析版（边想边做）")
    print("█" * 50)

    llm = HelloAgentsLLM()
    executor = ToolExecutor()
    executor.registerTool(
        "Search", "一个网页搜索引擎，用于查询实时或未知信息。", search
    )

    agent = ReActAgent(llm, executor, max_steps=5)
    agent.run(TEST_QUESTION)


def demo_react_v2():
    """演示 ReAct V2（Function Calling 版）：原生 tools API，零正则"""
    print("\n" + "█" * 50)
    print("█  演示 1b: ReAct V2 — Function Calling 版（零正则）")
    print("█" * 50)

    llm = HelloAgentsLLM()
    executor = ToolExecutor()
    executor.registerTool(
        "Search", "一个网页搜索引擎，用于查询实时或未知信息。", search
    )

    agent = ReActAgentV2(llm, executor, max_steps=5)
    agent.run(TEST_QUESTION)


def demo_compare():
    """对比演示：先跑正则版，再跑 Function Calling 版"""
    demo_react()
    print("\n\n" + "=" * 60)
    print("  以上是 ReAct（正则解析版），以下是 ReAct V2（Function Calling 版）")
    print("=" * 60)
    demo_react_v2()


def demo_plan_solve():
    """演示 Plan-and-Solve：先规划后执行"""
    print("\n" + "█" * 50)
    print("█  演示 2: Plan-and-Solve — 先规划后执行")
    print("█" * 50)

    llm = HelloAgentsLLM()
    agent = PlanAndSolveAgent(llm)
    agent.run(TEST_QUESTION)


def demo_reflection():
    """演示 Reflection：自我反思与迭代优化"""
    print("\n" + "█" * 50)
    print("█  演示 3: Reflection — 自我反思与迭代优化")
    print("█" * 50)

    llm = HelloAgentsLLM()
    agent = ReflectionAgent(llm, max_iterations=2)
    agent.run(TEST_QUESTION)


if __name__ == "__main__":
    import sys

    # 支持命令行选择演示
    demos = {
        "react": demo_react,
        "react2": demo_react_v2,
        "compare": demo_compare,
        "plan": demo_plan_solve,
        "reflection": demo_reflection,
        "all": lambda: (
            demo_react(),
            demo_plan_solve(),
            demo_reflection(),
        ),
    }

    if len(sys.argv) > 1 and sys.argv[1] in demos:
        demos[sys.argv[1]]()
    else:
        print("用法: python -m chapter4.main [react|react2|compare|plan|reflection|all]")
        print("  react     - ReAct 正则解析版")
        print("  react2    - ReAct Function Calling 版")
        print("  compare   - 对比两种 ReAct 实现")
        print("  plan      - Plan-and-Solve")
        print("  reflection- Reflection")
        print("  all       - 全部演示")
        print("\n默认运行 compare 对比演示\n")

        demo_compare()
