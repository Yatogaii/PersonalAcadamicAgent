1. 手动的写一个工具让Agent 汇报情况会显著的提升工作流程，比如 [CollectorProgressReport]。
2. LangChain 太不适合大型项目了，如果想要真正的多 Agent，需要把另一个 Agent 封装为一个 Tool 给主 Agent，非常不可控且不直观，远远不如 langgraph 的 State Driven 机制。