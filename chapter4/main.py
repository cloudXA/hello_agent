"""
第四章 智能体经典范式 — 演示入口

运行方式:
    cd Hello-Agents
    python -m chapter4.main

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
from chapter4.react_agent import ReActAgent
from chapter4.plan_solve_agent import PlanAndSolveAgent
from chapter4.reflection_agent import ReflectionAgent


def demo_react():
    """演示 ReAct：需要搜索工具的任务"""
    print("\n" + "█" * 50)
    print("█  演示 1: ReAct — 边想边做")
    print("█" * 50)

    llm = HelloAgentsLLM()

    executor = ToolExecutor()
    executor.registerTool(
        "Search", "一个网页搜索引擎，用于查询实时或未知信息。", search
    )

    agent = ReActAgent(llm, executor, max_steps=5)
    agent.run("哪些因为火灾失控如浏阳烟花的领导和市领导, 泉州鞋厂的领导和市领导后来受到处罚了吗")


def demo_plan_solve():
    """演示 Plan-and-Solve：多步数学推理"""
    print("\n" + "█" * 50)
    print("█  演示 2: Plan-and-Solve — 先规划后执行")
    print("█" * 50)

    llm = HelloAgentsLLM()
    agent = PlanAndSolveAgent(llm)
    agent.run(
        "哪些因为火灾失控如浏阳烟花的领导和市领导, 泉州鞋厂的领导和市领导后来受到处罚了吗"
    )


def demo_reflection():
    """演示 Reflection：代码优化"""
    print("\n" + "█" * 50)
    print("█  演示 3: Reflection — 自我反思与迭代优化")
    print("█" * 50)

    llm = HelloAgentsLLM()
    agent = ReflectionAgent(llm, max_iterations=2)
    agent.run("编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。")


if __name__ == "__main__":
    import sys

    # 支持命令行选择演示
    demos = {
        "react": demo_react,
        "plan": demo_plan_solve,
        "reflection": demo_reflection,
        "all": lambda: (demo_react(), demo_plan_solve(), demo_reflection()),
    }

    if len(sys.argv) > 1 and sys.argv[1] in demos:
        demos[sys.argv[1]]()
    else:
        print("用法: python -m chapter4.main [react|plan|reflection|all]")
        print("默认运行 Plan-and-Solve 演示（无需额外 API Key）\n")

        # 默认运行 Plan-and-Solve（不依赖外部 API Key）
        demo_plan_solve()
