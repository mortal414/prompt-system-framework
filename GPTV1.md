# Prompt System Framework – Project Plan (Draft)

This document is meant to **shape the idea**, surface assumptions, and list the questions that need answering before implementation.

---

## 1. Project Overview

**Goal:**  
Create a framework for designing and running **systems of prompting** – not just single prompts.

- A **system** is a graph of LLM calls (nodes) with defined inputs, outputs, and routing logic.
- From the outside, it behaves like a single chatbot or API:
  - You send one request (overall input).
  - It internally runs multiple prompts/nodes.
  - It returns a final, more reliable/structured response.
- Long-term, there will be a **UI** for visually building and editing these systems.
- As a starting point, it will likely be driven by a **CLI** and configuration files.

---

## 2. Core Goals

1. **Model multi-step reasoning explicitly**
   - Break complex behavior into smaller LLM nodes.
   - Make flows explicit instead of hidden inside one giant prompt.

2. **Make systems inspectable and debuggable**
   - See intermediate outputs.
   - Edit node prompts and templates in isolation.

3. **Make systems testable**
   - Node-level tests.
   - End-to-end system tests.
   - Easy regression checks when prompts change.

4. **Prepare for a visual builder**
   - Represent systems in a way that is easy for a future UI to render and manipulate (graph model).

---

## 3. Core Concepts

### 3.1 System

A **System** is:

- A set of nodes connected as a **graph** (DAG or graph with limited loops).
- A defined **overall input**:
  - What the user (or client) provides at the start.
- A defined **overall output**:
  - What the system returns when it finishes.
- A configuration for:
  - Models / providers.
  - Runtime options (logging, retries, etc., possibly later).

### 3.2 Node

A **Node** represents one LLM call plus its wiring.

Each node consists of:

- **Identity**
  - `id` (unique name in the system).
  - Optional `label` / description.

- **Instructions**
  - High-level behavior: what this node is supposed to do.

- **Prompt Template**
  - How to assemble the final prompt from its inputs.
  - Example (conceptual):

        You are a chess move validator.
        Position: {{position_fen}}
        Move: {{move_san}}
        Respond with JSON: {"is_valid": true/false, "reason": "..."}

- **Inputs**
  - Fields required for this node to run.
  - Values can come from:
    - Overall system input.
    - Outputs of earlier nodes.
    - Constants / config.

- **Output Schema**
  - Expected structure of the node’s response:
    - e.g., JSON with named fields, or structured text with parsing rules.
  - These fields may be used by downstream nodes or by the system’s final output.

- **Routing Rules (optional)**
  - Conditions based on output fields that determine:
    - Which node runs next.
    - Whether to retry, loop, or terminate.

**Conceptual node definition (for design thinking):**

    NODE:
      id: "validate_move"
      role: "Check if a proposed chess move is legal."

      INSTRUCTIONS:
        Given a chess board position (FEN) and a move (SAN), determine if the move is legal.

      PROMPT_TEMPLATE:
        Position: {{position_fen}}
        Move: {{move_san}}
        Respond with JSON:
        {
          "is_valid": true or false,
          "reason": "<short explanation>"
        }

      INPUTS:
        - from: system_input
          fields: [position_fen, move_san]

      OUTPUT_SCHEMA:
        - is_valid: boolean
        - reason: string

      ROUTING:
        if is_valid == true -> node: "accept_move"
        if is_valid == false -> node: "propose_new_move"
        
## 4. Capabilities (What the User Can Do)

In the final framework (CLI + eventual UI), the user should be able to:

- **Work with the graph**
  - Add nodes (`+`) to the system graph.
  - Remove nodes (`-`) from the system graph.
  - Rewire connections between nodes.

- **Design node logic**
  - Inspect and edit node instructions.
  - Inspect and edit prompt templates.
  - Define node input sources (which fields from where).
  - Define node output schema.

- **Define system I/O**
  - Declare **overall system input** fields (what the system expects at the start).
  - Declare **overall system output** fields (what the system returns at the end).
  - Map overall output fields to specific node outputs.

- **Run and inspect**
  - Execute a system with a given input.
  - View intermediate node outputs for debugging.
  - Run a single node in isolation with custom inputs.

- **Test**
  - Define test cases for individual nodes.
  - Define end-to-end test scenarios for the full system.

---

## 5. Execution Model

From the outside:

- User (or calling code) sends **one structured input**.
- The system runs its internal graph.
- Returns a **structured overall output**.

Internally:

1. The system starts with the **overall input context**.
2. Nodes become **runnable** when:
   - All required input fields are available.
   - Any conditions for running (e.g., flags) are satisfied.
3. When a node runs:
   - It assembles the prompt using its template.
   - Calls the LLM (via API).
   - Parses/validates the output against expected schema.
   - Stores the results in a shared context (accessible to downstream nodes).
4. Routing:
   - Simple linear sequences: Node A → Node B → Node C.
   - Conditional branches:
     - e.g., `if is_valid == false → Node "retry_move"`.
   - Optional loops with safeguards (max iterations).

Error handling ideas:

- What happens if:
  - The model returns malformed JSON?
  - Required fields are missing?
  - Output fails validation?
- Options:
  - Retry with same inputs (up to N times).
  - Route to a fallback node.
  - Abort with structured error.

---

## 6. Testing & Quality

Testing is a **first-class feature**.

### 6.1 Node-Level Tests

- Each node can have test cases:
  - Input values.
  - Expected output patterns or constraints.
- Example (chess validator node):
  - Input: legal move → `is_valid` is `true`.
  - Input: illegal move → `is_valid` is `false`.

### 6.2 System-Level Tests

- Provide a complete system input and assert:
  - Final overall output (shape + key properties).
  - Optionally: intermediate node outputs (for debugging).

### 6.3 Regression / Snapshot Testing

- Record outputs for known scenarios.
- Detect when prompt changes cause unexpected behavior shifts.

### 6.4 Metrics (optional but desirable)

- Track per-run:
  - Node execution order.
  - Token usage / cost.
  - Latency per node.
- Useful for optimization and debugging.

---

## 7. Example: Chess Pipeline (Sketch)

**High-level system behavior:**

1. Receive a chess position and some context.
2. Propose a move.
3. Validate the move.
4. If invalid, loop back and propose a new move with feedback.
5. Return a valid move and explanation.

Possible nodes:

- `propose_move`
  - Inputs: board position, game context.
  - Output: `move_san`, `rationale`.

- `validate_move`
  - Inputs: `position_fen`, `move_san`.
  - Output: `is_valid`, `reason`.

- `handle_invalid_move` (optional)
  - Inputs: `reason`, previous `move_san`.
  - Output: new instructions/hints to feed back into `propose_move`.

- `final_response`
  - Inputs: final valid move, combined rationale.
  - Output: final system output to the user.

We can **test** `validate_move` in isolation with a suite of known legal/illegal moves, independent of the rest of the system.

---

## 8. Open Design Questions (To Answer Before/While Designing)

These are the **important unknowns** you’ll want to clarify.

### 8.1 Primary Use Cases

- What are the **first concrete use cases** you want to support?
  - Chess pipelines?
  - Debate/argument builders?
  - Code review flows?
  - Data cleaning / classification chains?
- Are you optimizing for:
  - **Your own experimentation and research**, or
  - A **general-purpose framework** that others might use?

### 8.2 System Definition Format

- How should systems be defined and stored?
  - YAML/JSON configuration files?
  - Python code (class/DSL)?
  - Both (e.g., config + optional Python hooks)?
- Do you want to be able to:
  - Generate configs from a CLI wizard?
  - Edit configs directly in a text editor?
  - Both?

### 8.3 Node I/O and Type System

- How strict should **node outputs** be?
  - Must every node return **valid JSON** matching a schema?
  - Is “best-effort JSON with validation + fallback” acceptable?
  - Do you want to support arbitrary text + parsing rules?

- What types should the framework support out of the box?
  - Simple primitives: string, number, boolean.
  - Arrays / lists.
  - Nested objects.
  - Enums (e.g., `"ACCEPT" | "REJECT" | "RETRY"`).

- How are **types and schemas** specified?
  - Simple ad-hoc notation in YAML?
  - JSON Schema?
  - Python dataclasses / Pydantic models?

### 8.4 Models & Providers

- Do you want:
  - One global model/provider per system?
  - Per-node model selection?
- Are there constraints on:
  - API providers (OpenAI only? multiple providers)?
  - Rate limits or cost ceilings per run?

### 8.5 Control Flow Complexity

- What level of branching complexity do you want to support initially?
  - Simple `if/else` based on boolean fields?
  - Comparisons: `score > threshold`, string matching, etc.?
  - More complex expressions?

- Do you want **loops** (e.g., retry until success) baked in?
  - If yes, how to constrain them?
    - Max iterations per loop?
    - Manual stop conditions?

- How should **errors** be treated by default?
  - Immediate failure?
  - Retry logic (with backoff)?
  - Special “error handler” nodes?

### 8.6 Testing Strategy

- How do you want to **author tests**?
  - As YAML test files?
  - As Python test code (e.g., `pytest`)?
  - A mixture?

- Minimum testing features you want:
  - Is node-level testing mandatory?
  - Is end-to-end system testing mandatory?
  - Do you want snapshot tests (store outputs and detect changes)?

- Do you want to track **token usage and cost** as part of tests?

### 8.7 Execution & Observability

- How should runs be **logged**?
  - Plain-text logs?
  - Structured logs (JSON lines per node)?
- Do you want the ability to:
  - Replay a previous run with the same inputs?
  - Compare runs (before vs after prompt changes)?

- What metadata should be stored per run?
  - System definition version?
  - Node versions?
  - Model identifiers & temperatures?

### 8.8 Future UI Considerations

- Should the UI operate directly on the same **system definition files** used by the CLI?
- What UI operations are essential?
  - Drag-and-drop nodes?
  - Edit node templates inline?
  - Visualize execution traces over the graph?
- Do you want the UI to:
  - Trigger and monitor runs?
  - Visualize test results per node / per system?

---

## 9. Next Steps for Clarifying the Idea

When you’re ready, you can:

1. **Pick one main use case** (e.g., chess move pipeline or a simple validation system).
2. Design a **very small system** (2–3 nodes) end to end:
   - System input.
   - Node definitions (instructions, templates, I/O).
   - System output.
   - A few test cases.
3. Use that as a **concrete example** to refine:
   - Node schema.
   - Test format.
   - Control flow and error handling.

That small example will make many of the open questions easier to answer.
