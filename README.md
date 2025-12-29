# AI Agent Frameworks Research

**Comprehensive comparison of 64 AI agent and LLM orchestration frameworks**

Last Updated: December 2025

---

## Quick Navigation

| Resource | Description | Best For |
|----------|-------------|----------|
| [Dashboard (HTML)](./ai-agent-frameworks-dashboard.html) | Interactive filterable/sortable table with decision tree | Visual exploration, filtering by criteria |
| [Pricing Calculator (HTML)](./ai-agent-pricing-calculator.html) | LLM cost calculator with 25+ models | Budget planning, model cost comparison |
| [Comparison Spreadsheet (CSV)](./ai-agent-frameworks-comparison.csv) | Raw data: 64 frameworks x 18 columns | Excel/Sheets analysis, custom filtering |
| [Deep Dive Guide (MD)](./ai-agent-frameworks-deep-dive.md) | Detailed analysis of top 15 frameworks | Learning framework internals, code examples |
| [Getting Started Guide (MD)](./ai-agent-getting-started-guide.md) | Beginner's guide from first agent to production | New to AI agents, learning path |

---

## TL;DR Decision Tree

```
What are you building?
│
├─ Simple chatbot or RAG app
│  ├─ Quick start → LangChain or LlamaIndex
│  ├─ Enterprise RAG → RAGFlow (70k stars!)
│  └─ Desktop/local → AnythingLLM (53k stars)
│
├─ Multi-agent collaboration
│  ├─ Role-based teams → CrewAI
│  ├─ AI software company → MetaGPT (62k stars!)
│  ├─ Conversation-based → AutoGen/AG2
│  ├─ Fast & lightweight → Agno
│  └─ Distributed agents → AgentScope (Alibaba)
│
├─ Production enterprise app
│  ├─ Microsoft/.NET shop → Semantic Kernel
│  ├─ Google Cloud → Google ADK or Vertex AI
│  ├─ AWS → Bedrock Agents or Strands SDK
│  ├─ OpenAI-only → OpenAI Agents SDK
│  └─ Microsoft 365/Teams → Microsoft 365 Agents
│
├─ Coding assistant
│  ├─ Terminal/CLI → Aider or Gemini CLI (89k stars!)
│  ├─ Full IDE experience → OpenHands
│  └─ GitHub issue fixing → SWE-agent (Princeton)
│
├─ Data & Web
│  ├─ Web scraping for AI → Firecrawl (72k stars!)
│  ├─ Browser automation → BrowserUse (74k stars!)
│  └─ GraphRAG UI → Kotaemon (25k stars)
│
├─ Infrastructure & Memory
│  ├─ Memory layer → Mem0 (45k stars)
│  ├─ Secure code sandbox → Daytona (41k stars)
│  └─ Tool integrations → Composio (26k stars)
│
├─ Visual/low-code
│  ├─ Technical users → LangFlow
│  ├─ Business automation → n8n or Lindy
│  └─ AI workflow automation → Activepieces (20k stars)
│
├─ Specialized
│  ├─ Deep research → DeerFlow (ByteDance)
│  ├─ Customer service → Parlant (17k stars)
│  ├─ State machine agents → Burr (Apache)
│  └─ Stateful sessions → Julep
│
└─ Research/experimental
   └─ DSPy, CAMEL-AI, or PocketFlow (minimalist)
```

---

## Top 10 Frameworks by GitHub Stars

| Rank | Framework | Stars | Category |
|------|-----------|-------|----------|
| 1 | **LangChain** | 122k | Core Orchestration |
| 2 | **Gemini CLI** | 89k | Coding Agent |
| 3 | **BrowserUse** | 74k | Browser Automation |
| 4 | **Firecrawl** | 72k | Web Scraping |
| 5 | **RAGFlow** | 71k | RAG Engine |
| 6 | **OpenHands** | 66k | Coding Agent |
| 7 | **MetaGPT** | 62k | Multi-Agent |
| 8 | **AutoGen** | 53k | Multi-Agent |
| 9 | **AnythingLLM** | 53k | RAG Platform |
| 10 | **LangFlow** | 50k | Visual/Low-Code |

---

## Framework Categories

### Core Orchestration (General Purpose)
| Framework | Stars | Best For | Risk |
|-----------|-------|----------|------|
| **LangChain** | 122k | General orchestration, RAG, prototyping | Low |
| **LangGraph** | 22k | Stateful multi-agent, complex workflows | Low |
| **Semantic Kernel** | 27k | Enterprise .NET, Microsoft ecosystem | Low |
| **Google ADK** | 17k | Google Cloud, Gemini models | Low |
| **OpenAI Agents SDK** | N/A | OpenAI-native development | Low |

### Multi-Agent Systems
| Framework | Stars | Best For | Risk |
|-----------|-------|----------|------|
| **MetaGPT** | 62k | AI software company, code generation | Medium |
| **AutoGen** | 53k | Research, multi-agent conversation | Medium |
| **CrewAI** | 42k | Role-based teams, workflows | Medium |
| **Agno** | 36k | Fast multi-agent, Teams, AgentOS | Medium |
| **OpenAI Swarm** | 21k | Educational (DEPRECATED) | High |
| **Letta** | 20k | Persistent memory agents | Medium |
| **AgentScope** | 15k | Distributed agents (Alibaba) | Low |
| **Langroid** | 4k | Multi-agent RAG, SQL/document chat | Low |

### RAG & Data Platforms
| Framework | Stars | Best For | Risk |
|-----------|-------|----------|------|
| **RAGFlow** | 71k | Enterprise RAG, GraphRAG | Low |
| **AnythingLLM** | 53k | Desktop RAG, local LLMs | Low |
| **LlamaIndex** | 46k | Data ingestion, 160+ connectors | Low |
| **Kotaemon** | 25k | RAG UI, GraphRAG, self-hostable | Low |
| **Haystack** | 24k | Enterprise RAG, document QA | Low |

### Coding Agents
| Framework | Stars | Best For | Risk |
|-----------|-------|----------|------|
| **Gemini CLI** | 89k | Terminal AI, Gemini-powered | Low |
| **OpenHands** | 66k | Full dev platform, IDE | Low |
| **Aider** | 39k | Terminal pair programming, Git | Low |
| **SWE-agent** | 18k | GitHub issue fixing (Princeton) | Medium |

### Web & Browser
| Framework | Stars | Best For | Risk |
|-----------|-------|----------|------|
| **BrowserUse** | 74k | Web automation, form filling | Low |
| **Firecrawl** | 72k | Web scraping for AI, LLM-ready | Low |

### Infrastructure & Memory
| Framework | Stars | Best For | Risk |
|-----------|-------|----------|------|
| **Mem0** | 45k | Memory layer for AI agents | Low |
| **Daytona** | 41k | Secure code execution sandbox | Low |
| **Ray** | 35k | Distributed agent execution | Low |
| **Composio** | 26k | 100+ tool integrations | Low |

### Visual/Workflow Automation
| Framework | Stars | Best For | Risk |
|-----------|-------|----------|------|
| **LangFlow** | 50k | Visual prototyping, drag-and-drop | Low |
| **Activepieces** | 20k | AI workflow, 400 MCP servers | Low |
| **n8n** | N/A | Workflow automation, 500+ integrations | Low |
| **Lindy** | N/A | Business automation, non-technical | Low |

### Cloud Platforms (Managed)
| Platform | Provider | Best For |
|----------|----------|----------|
| **Bedrock Agents** | AWS | AWS-native managed agents |
| **Strands SDK** | AWS | AWS ecosystem, MCP support |
| **Azure AI Foundry** | Microsoft | Azure-native enterprise |
| **Vertex AI Agents** | Google | GCP-native, A2A support |

### Enterprise Platforms
| Framework | Stars | Best For | Risk |
|-----------|-------|----------|------|
| **Salesforce Agentforce** | N/A | Enterprise CRM, sales/service | Low |
| **Microsoft 365 Agents** | 630 | Teams, Copilot extensions | Low |
| **BeeAI Framework** | 3k | Enterprise multi-agent, A2A | Low |

### Specialized
| Framework | Stars | Best For | Risk |
|-----------|-------|----------|------|
| **DeerFlow** | 19k | Deep research (ByteDance) | Medium |
| **Suna** | 19k | General agent platform | Medium |
| **Parlant** | 17k | Customer service agents | Low |
| **PocketFlow** | 9k | Minimalist 100-line framework | Medium |
| **Julep** | 7k | Stateful agents with sessions | Medium |
| **Burr** | 2k | State machine agents (Apache) | Low |

### Protocols
| Protocol | Owner | Purpose |
|----------|-------|---------|
| **MCP** | Anthropic | Agent-to-tool connection standard |
| **A2A** | Linux Foundation | Agent-to-agent communication |

---

## Quick Start Recommendations

### For Beginners
1. **Start with LangChain** - Best docs, largest community, most examples
2. **Read the [Getting Started Guide](./ai-agent-getting-started-guide.md)** - Covers first agent to production
3. **Use the [Dashboard](./ai-agent-frameworks-dashboard.html)** - Filter by your criteria

### For Production Teams
1. **Check [Pricing Calculator](./ai-agent-pricing-calculator.html)** - Estimate costs early
2. **Review "Production Ready" column** in dashboard
3. **Consider vendor lock-in** - See "Risk Level" in comparison

### For Researchers
1. **DSPy** for prompt optimization
2. **SWE-agent** for code generation research
3. **MetaGPT** for multi-agent software development
4. **CAMEL-AI** for multi-agent research

---

## Key Insights from Research

### Protocol Support (MCP & A2A)
- **MCP (Model Context Protocol)**: LangChain, LangGraph, PydanticAI, Mastra, Smolagents, Google ADK, Semantic Kernel, AWS Strands, BrowserUse, Langroid, BeeAI, Gemini CLI, Firecrawl, RAGFlow, AnythingLLM, Composio, Activepieces
- **A2A (Agent-to-Agent)**: Google ADK, Vertex AI, LangChain, LangGraph, PydanticAI, CrewAI, BeeAI Framework

### Funding & Stability
| Tier | Frameworks |
|------|------------|
| **Big Tech Backed** | Semantic Kernel, Google ADK, Gemini CLI, OpenAI Agents SDK, AWS Strands, BeeAI (IBM), DeerFlow (ByteDance), AgentScope (Alibaba), Microsoft 365 Agents |
| **Foundation Backed** | Burr (Apache), A2A (Linux Foundation) |
| **VC Unicorn** | LangChain ($125M), CrewAI, Haystack ($18M), BrowserUse, Firecrawl, RAGFlow, Mem0, Daytona |
| **Growing** | PydanticAI, Agno, Mastra, Letta ($10M), Julep, MetaGPT, AnythingLLM |

### Common Limitations
- **LangChain**: Over-engineered for simple tasks, API churn
- **MetaGPT**: High token costs, complex orchestration
- **AutoGen**: Fork drama (AG2), experimental status
- **OpenAI Swarm**: DEPRECATED - use OpenAI Agents SDK instead
- **ControlFlow**: ARCHIVED - merged into Marvin
- **Julep**: Cloud-first, self-hosting can be complex

---

## Pricing Overview (LLM Models)

See [Pricing Calculator](./ai-agent-pricing-calculator.html) for interactive comparison.

### Budget Tiers (per 1M tokens)

| Tier | Input | Output | Models |
|------|-------|--------|--------|
| **Ultra Budget** | <$0.10 | <$0.50 | Groq Llama, Cerebras, DeepSeek |
| **Budget** | $0.10-0.50 | $0.50-2.00 | GPT-4o-mini, Claude Haiku, Gemini Flash |
| **Standard** | $1-5 | $5-15 | GPT-4o, Claude Sonnet, Gemini Pro |
| **Premium** | $10-15 | $30-75 | GPT-4.5, Claude Opus, o1 reasoning |

---

## Files in This Repository

```
ai-frameworks-research/
├── README.md                              # This file
├── ai-agent-frameworks-comparison.csv     # Raw data (64 frameworks x 18 columns)
├── ai-agent-frameworks-dashboard.html     # Interactive dashboard
├── ai-agent-pricing-calculator.html       # LLM cost calculator
├── ai-agent-frameworks-deep-dive.md       # Detailed framework analysis
└── ai-agent-getting-started-guide.md      # Beginner's learning path
```

---

## Framework Count by Category

| Category | Count |
|----------|-------|
| Core Orchestration | 6 |
| Multi-Agent | 9 |
| RAG & Data | 5 |
| Coding Agent | 4 |
| Web/Browser | 2 |
| Infrastructure/Memory | 4 |
| Visual/Workflow | 4 |
| Cloud Platform | 4 |
| Enterprise Platform | 3 |
| Specialized | 6 |
| Lightweight | 7 |
| Utility | 4 |
| Protocol | 2 |
| Research | 2 |
| Voice/Realtime | 1 |
| Observability | 1 |
| **Total** | **64** |

---

## Research Methodology

### Sources Consulted
- Official documentation for all 64 frameworks
- GitHub repositories (stars, issues, activity)
- Benchmarks: SWE-bench, GAIA, AgentBench, WebArena
- Community: Reddit r/LocalLLaMA, r/MachineLearning, Hacker News
- Enterprise case studies and production deployments

### Criteria Evaluated
1. **Technical**: Language support, MCP/A2A, function calling, multimodal
2. **Maturity**: Production ready, GitHub stars, funding, community size
3. **Risk**: Vendor lock-in, license, maintenance activity
4. **Usability**: Setup time, documentation quality, learning curve

---

## Contributing

This research is open source! To suggest updates:
1. Check if the framework/information is already covered
2. Provide sources for any new data
3. Focus on production-relevant information
4. Open an issue or PR at [github.com/ShreeMulay/ai-agent-frameworks-research](https://github.com/ShreeMulay/ai-agent-frameworks-research)

---

## License

MIT License - Feel free to use this research for your own projects!

Research compiled from public documentation and repositories.
