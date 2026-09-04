---
title: Don't Let Your AI Go Rogue
excerpt: Learn how to keep autonomous AI agents safe and predictable by adding simple human approval checkpoints.
date: 2026-09-04
bannerImage: 8-0-hitl-architecture.png
readTime: 9 min read
tags: AI, LangChain, LangGraph, Python, Architecture, PostgreSQL, HITL, Human-in-the-Loop
---

![HITL Architecture Hero](images/8-1-hitl-architecture-hero.png)  

## Hey there, fellow AI system builder! 🚀

As an engineer building with LLMs, you are likely already familiar with orchestrating autonomous AI Agents to execute tool calls, query databases, and trigger external APIs automatically. However, running agentic workflows on total autopilot in production quickly introduces some critical systems challenges.
  
When an agent executes state-mutating actions without guardrails, a small hallucination or misinterpreted parameter can alter production databases and cause unrecoverable side effects across your infrastructure.
  
This is precisely why **Human-in-the-Loop (HITL)** architecture has become an essential engineering pattern for production-grade agentic pipelines. Think of HITL as putting a red stop sign in front of AI Agent. Before it does anything risky, it pauses, asks a human for approval, and waits for your thumbs-up.   
  
Today, we are going to explore how to implement these safe human review checkpoints directly into your production pipelines!

But before we jump into the code, let's break down the core architectural concepts step by step.

---

## 1. What is Human-in-the-Loop (HITL)?

In plain words, **Human-in-the-Loop** is a design setup where an automated AI pipeline pauses at specific checkpoints. It freezes its current progress, hands control over to a human operator, and patiently waits for approval or feedback before resuming.

For engineers, think of HITL as a **durable, non-blocking breakpoint combined with state saving**.

![HITL Architecture Flow](images/8-2-hitl-workflow.png)  

When an AI agent reaches a predefined stop sign (an interrupt checkpoint):

* The application pauses processing immediately.
* The current progress and memory are saved safely into a database like PostgreSQL or Redis.
* The server process stops the run completely, meaning zero background threads or workers sit idle or waste system RAM.
* When a human approves or gives feedback through a UI or API call, the system reloads the saved state back from storage and picks up right where it left off!

---

## 2. Why HITL Matters: The Core Engineering Challenges

Running pure, unsupervised AI agents introduces three major risks to your application:

* **The Blast Radius Problem (Side Effects):** Safe actions (like searching a document or reading logs) carry low risk. But state-changing actions such as running `UPDATE` or `DELETE` SQL queries, charging a card via `stripe.charge()`, or modifying cloud infrastructure have a massive blast radius. Once executed, you cannot easily undo them without manual fixes.
* **Hallucinations & Ambiguity:** When given vague instructions or missing parameters, autonomous AI often makes up facts rather than ask for help. HITL lets the agent pause and ask a human for clarification when confidence is low.
* **Continuous Alignment:** For multi-step work (like drafting legal documents or generating code), early mistakes compound quickly. Adding a human review step early on prevents tiny errors from snowballing downstream!

Now let me show you the three simple patterns used to build these checkpoints.

---

## 3. Core Architectural Patterns of HITL

When building human interaction into your workflows, three main execution patterns cover almost every use case:

![HITL Patterns](images/8-3-hitl-patterns.png)  

* **Pattern 1: Approval Gatekeepers:** The agent plans an action (like `cancel_subscription(user_id=8923)`). The pipeline pauses and shows the plan on a dashboard. If approved, it runs; if rejected, it stops or triggers a backup plan.
* **Pattern 2: Edit and Override:** The agent drafts an output (like a SQL query or email draft). Before running, the payload opens in an editable screen where a human can tune the parameters directly before sending it off.
* **Pattern 3: Interactive Feedback Loops:** A human provides direct feedback (like *"Make this sound less formal"*). The graph feeds this comment back into the AI context window so it can revise its draft.

---

## 4. Implementating HITL

Today we will implement a simple HITL workflow in **Langchain** and **LangGraph**, but you can use the same patterns with any LLM framework of your choice. The concept remains the same.

So we will use **LangChain** (a popular software toolkit that connects LLMs to external data and actions) alongside **LangGraph** (a framework built inside LangChain specifically designed to manage complex, step-by-step stateful workflows).

LangGraph relies on three core concepts to drive HITL:

* **`interrupt()`:** Calling `interrupt(payload)` inside a graph step pauses execution immediately, sends the `payload` data to the client UI, and saves state to storage.
* **`Command(resume=payload)`:** To wake up a paused execution graph, you call the graph using a `Command` object containing the human's response.
* **Checkpointers (`PostgresSaver`, `RedisSaver`):** Checkpointers are the backend engines that save and reload application state from persistent databases using a unique `thread_id`. 💡

Let's check out a couple of crucial operational mechanics before writing code!

---

## 5. Critical Technical Considerations

Before jumping into code, here are two major mechanics every developer needs to understand:

### Node Re-execution and Idempotency

When an interrupted step wakes back up using `Command(resume=payload)`, LangGraph runs that step's function **from top again**. Once it reaches the `interrupt()` line, it skips the pause, loads the stored `payload` right into the variable, and continues moving downward!

```python
# ❌ DANGEROUS ANTI-PATTERN: Side effects before interrupt() will re-run on resume
def flawed_node(state):
    db.logs.insert("Review initiated")  # ⚠️ RUNS TWICE (Initial run + Resume)
    
    response = interrupt({"draft": state["draft"]})
    return {"status": "complete"}

# ✅ PRODUCTION PATTERN: Place non-idempotent side effects AFTER interrupt()
def safe_node(state):
    # Pure data processing/reads go before interrupt
    draft = state["draft"]
    
    response = interrupt({"draft": draft})
    
    # Side-effects execute strictly AFTER resumption
    db.logs.insert("Review approved") 
    return {"status": "complete"}

```

### Persistence: Moving from Dev to Production

While `InMemorySaver` is convenient for quick local testing, it stores application memory in local RAM. If your server restarts or redeploys, all paused sessions disappear! Production builds must use a durable database checkpointer like **`PostgresSaver`** or **`RedisSaver`**.

---

## 6. End-to-End HITL Code Example

Let's build a **Code Review & Approval Loop** using `PostgresSaver` durability and safe idempotency patterns!

### Prerequisites

Run this command in your terminal to install the required packages:

```bash
pip install langgraph langgraph-checkpoint-postgres psycopg[binary] psycopg-pool uuid
```

or if you use UV, then

```bash
uv add langchain langgraph langgraph-checkpoint-postgres psycopg[binary] psycopg-pool uuid
```

Start a local PostgreSQL database container which will be used as the checkpoint database:

```bash
docker run --name langgraph-postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres
```

or if you prefer podman then 

```bash
podman run --name langgraph-postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d docker.io/library/postgres
```

### Complete Python Implementation

Here is the exact code to set up, pause, edit, and resume your graph workflow:

```python
import uuid
from typing import TypedDict, Literal, Optional, Dict, Any
from langgraph.graph.state import CompiledStateGraph
from psycopg_pool import ConnectionPool
from psycopg import connect
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import interrupt, Command

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"

# =====================================================================
# 1. Define Application State
# =====================================================================
class CodeReviewState(TypedDict):
    task_description: str
    code_draft: str
    review_status: str

# =====================================================================
# 2. Define Graph Nodes
# =====================================================================
def draft_code_node(state: CodeReviewState):
    print("\n[Node: draft_code_node] Generating code solution...")
    task = state["task_description"]
    # Simulating LLM code generation
    return {
        "code_draft": f"# Auto-generated code for: {task}\ndef execute_task():\n    print('Task completed safely')"
    }

def human_review_node(state: CodeReviewState) -> Command[Literal["deploy_node", "draft_code_node"]]:
    print("\n[Node: human_review_node] Preparing checkpoint for human review...")

    current_draft = state["code_draft"]

    # Execution halts here on initial run / node entry.
    # On resumption via Command(resume=...), execution skips pause and assigns resume payload to 'human_response'.
    human_response = interrupt({
        "action": "code_review_required",
        "code_to_review": current_draft
    })

    print(f"\n[Node: human_review_node] Resumed execution with payload: {human_response}")

    decision = human_response.get("decision")

    if decision == "approve":
        return Command(
            update={"review_status": "approved"},
            goto="deploy_node"
        )
    else:
        feedback = human_response.get("feedback", "No specific feedback provided.")
        return Command(
            update={"review_status": f"rejected: {feedback}"},
            goto="draft_code_node"
        )

def deploy_node(state: CodeReviewState):
    print("\n[Node: deploy_node] Executing non-idempotent deployment operation...")
    return {"review_status": "deployed_to_production"}

# =====================================================================
# 3. Assemble Graph Topology
# =====================================================================
def build_agent_executor(connection_pool) -> CompiledStateGraph[CodeReviewState, Any, Any, Any]:
    builder = StateGraph(CodeReviewState)

    builder.add_node("draft_code_node", draft_code_node)
    builder.add_node("human_review_node", human_review_node)
    builder.add_node("deploy_node", deploy_node)

    builder.add_edge(START, "draft_code_node")
    builder.add_edge("draft_code_node", "human_review_node")
    builder.add_edge("deploy_node", END)

    checkpointer = PostgresSaver(connection_pool)
    checkpointer.setup()

    return builder.compile(checkpointer=checkpointer)

# =====================================================================
# 4. Agent Execution Wrapper Function
# =====================================================================
def run_agent(
    thread_id: str,
    input_payload: Optional[Dict[str, Any]] = None,
    resume_payload: Optional[Dict[str, Any]] = None
):
    """
    Manages DB connections, builds the app, and executes or resumes graph workflow.
    """
    # Run DDL (CREATE INDEX CONCURRENTLY) outside a transaction block
    with connect(DB_URI, autocommit=True) as standalone_conn:
        PostgresSaver(standalone_conn).setup()

    # Build Agent
    with ConnectionPool(conninfo=DB_URI, max_size=10) as pool:
        app = build_agent_executor(pool)

        config = RunnableConfig(configurable={"thread_id": thread_id})

        if resume_payload:
            print(f"\n>>> Resuming thread '{thread_id}' with payload: {resume_payload}")
            stream_input = Command(resume=resume_payload)
        else:
            print(f"\n>>> Starting thread '{thread_id}' with initial task...")
            stream_input = input_payload

        # Stream execution until complete or next interrupt point
        for event in app.stream(stream_input, config=config):
            print("Event emitted:", event)

        # Inspect and return current state snapshot
        return app.get_state(config)

# =====================================================================
# 5. Lean CLI Interactive Main Section
# =====================================================================
if __name__ == '__main__':
    thread_id = f"review-session-{uuid.uuid4().hex[:6]}"
    initial_input = {"task_description": "Create automated schema migration script"}

    # Initial Run
    snapshot = run_agent(thread_id, input_payload=initial_input)

    # Interactive Human-in-the-Loop CLI execution
    while snapshot.next:
        print("\n" + "="*60)
        print("⏸️  GRAPH INTERRUPTED: Awaiting Human Intervention")
        print("="*60)

        if snapshot.tasks and snapshot.tasks[0].interrupts:
            interrupt_data = snapshot.tasks[0].interrupts[0].value
            print(f"Action Required : {interrupt_data.get('action')}")
            print(f"Draft Code:\n{interrupt_data.get('code_to_review')}")

        print("-" * 60)
        user_choice = input("Enter decision ('approve' / 'reject'): ").strip().lower()

        if user_choice == "approve":
            resume_data = {"decision": "approve"}
        elif user_choice == "reject":
            feedback = input("Enter feedback for re-drafting: ").strip()
            resume_data = {"decision": "reject", "feedback": feedback or "Needs revision."}
        else:
            print("Invalid choice! Please type 'approve' or 'reject'.")
            continue

        # Resume Agent Execution
        snapshot = run_agent(thread_id, resume_payload=resume_data)

    # Pipeline Complete
    print("\n" + "="*60)
    print("🎉 PIPELINE EXECUTED TO COMPLETION")
    print("="*60)
    print("Final Status:", snapshot.values.get("review_status"))

```

### Run the code

```bash
python human_in_the_loop.py
```

or using UV

```bash
uv run human_in_the_loop.py
```

---

## 7. Let's Decipher The Code Step By Step

### **State Schema Definition**

```python
class CodeReviewState(TypedDict):
    ...
```

A stateful workflow requires a strictly defined runtime schema. `CodeReviewState` acts as the single source of truth passed across node boundaries. It holds the core task parameters, generated code artifacts, and pipeline review statuses.

### **Node Logic & Interrupt Mechanics**

```python
def draft_code_node(state: CodeReviewState):
    ...
```

The `draft_code_node` represents an LLM tool-calling phase. It accepts the input payload, generates a code artifact, and updates the `code_draft` state key before passing execution to the human gatekeeper.

```python
def human_review_node(state: CodeReviewState) -> Command[Literal["deploy_node", "draft_code_node"]]:
    ...
```

The `human_review_node` implements the core Human-in-the-Loop (HITL) pattern:

* **`interrupt(...)`:** Automatically freezes graph execution upon entry and surfaces the review payload (`code_to_review`). The current state snapshot is written to PostgreSQL.
* **Resumption:** When the pipeline resumes via `Command(resume=...)`, execution unblocks immediately after `interrupt()`, populating `human_response` with the operator's payload.
* **Dynamic Routing with `Command`:** Eliminates conditional edge boilerplate. If approved, the state updates to `"approved"` and routes directly to `deploy_node`. If rejected, it updates state with the feedback and loops back to `draft_code_node` for revision.

```python
def deploy_node(state: CodeReviewState):
    ...
```

The isolated terminal node for non-idempotent side effects (such as deploying code, executing SQL migrations, or firing external webhooks). This node can only be reached if explicitly cleared by a human approval in `human_review_node`.

### **Topology Assembly & Durable Checkpointing**

```python
def build_agent_executor(connection_pool) -> CompiledStateGraph[CodeReviewState, Any, Any, Any]:
    ...
```

The graph topology maps node sequences and state persistence:

* **Edges:** Standard edges link `START` -> `draft_code_node` -> `human_review_node`. Conditional transitions are handled dynamically by `Command` inside `human_review_node`.
* **Postgres Checkpointer:** Binds the graph to PostgreSQL. State, thread history, and active interrupts are persisted to disk, allowing the application process to shut down or restart safely while waiting for human intervention.

### **Execution Engine & Schema Initialization**

```python
def run_agent(
    thread_id: str,
    input_payload: Optional[Dict[str, Any]] = None,
    resume_payload: Optional[Dict[str, Any]] = None
):
    ...
```

`run_agent` isolates database lifecycle and execution dispatching:

* **Session Persistence:** Configures `thread_id` via `RunnableConfig` to isolate execution states across concurrent user sessions.
* **Execution vs Resumption:** If `resume_payload` is present, it passes `Command(resume=...)` to clear the active interrupt; otherwise, it initiates a new run with `input_payload`.
  
### **Interactive CLI Execution Loop**

```python
if __name__ == '__main__':
    ...
```

The execution loop simulates real-world human feedback:

1. **Initial Trigger:** Starts the pipeline. Execution automatically stops at `human_review_node`, saving state to Postgres.
2. **State Inspection:** `snapshot.next` detects the paused state. `snapshot.tasks[0].interrupts` exposes the exact code draft needing review.
3. **CLI Intervention:** Accepts manual operator approval or feedback (`input()`).
4. **Resumption Loop:** Invokes `run_agent` with the human payload. Rejection loops back to code generation; approval proceeds to production deployment and terminates the loop.

---

## 8. Wrap-Up & Conclusion

Integrating **Human-in-the-Loop** into your LLM agent architecture gives you the absolute best of both worlds: the speed and intelligence of AI combined with the safety and common sense of real human oversight. By pairing LangGraph's pause-and-resume mechanics with a reliable database like PostgreSQL, you can build powerful agents that you can actually trust in production environments!

Now it is your turn to jump in, try out the code pattern above, and build safer AI apps for your users!

Until next time, *Happy Coding!* 💻
