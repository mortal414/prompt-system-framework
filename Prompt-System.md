A **system of prompts** is a way of using an LLM as a *multi-step process* instead of a single question-and-answer.

In normal prompting, you ask one big question in natural language and get one big answer. That works for small tasks, but it breaks down when:

* the task is too complex to fit cleanly in a single prompt, or
* long, overloaded prompts cause the model's accuracy to drop, even if they're technically within the context window.

A system of prompts fixes this by **breaking work into explicit stages** and wiring them together:

* The overall task is decomposed into smaller **subtasks**, each handled by its own prompt.
* Each step has **clear inputs and outputs** (e.g., "take this position, output a move candidate", "take this move, output a legality judgment").
* **Later prompts consume the outputs of earlier ones**, forming a pipeline or graph instead of a single call.
* The system can implement **control flow**: retries, branches, or loops. For example, if a step returns an invalid result `Y` when we needed `Z`, we can automatically re-prompt with `X` again (optionally including `Y` as context) until the output passes validation.

In other words, a system of prompts treats LLM calls like nodes in a workflow: each node does one thing well, and the overall behavior emerges from how those nodes are sequenced, validated, and looped--rather than trying to force everything into one giant prompt.
