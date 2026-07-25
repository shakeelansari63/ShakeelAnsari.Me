# Tool Schema Design — Naming, Descriptions, Parameter Constraints

> A correct tool fails silently when the model cannot tell when to use it. Naming, descriptions, and parameter shapes drive 10 to 20 percentage-point swings in tool-selection accuracy.

**Type:** Learn
**Languages:** Python (stdlib, tool schema linter)
**Prerequisites:** Phase 13 · 01, Phase 13 · 04
**Time:** ~45 minutes

## Learning Objectives
- Write a tool description using the "Use when X. Do not use for Y." pattern
- Name tools in stable `snake_case` across a large registry
- Choose between atomic tools and monolithic tools
- Run a tool-schema linter against a registry and fix the findings

## The Problem

An agent with 30 tools: wrong tool picked because descriptions are ambiguous, or no tool picked when one fits. Composio's 2025 field guide measured 10-20 percentage-point accuracy swings from renaming and rewriting descriptions alone.

## The Concept

### Naming rules

1. `snake_case` — every provider's tokenizer handles it cleanly.
2. Verb-noun order: `get_weather`, not `weather_get`.
3. No tense markers: `get_weather`, not `got_weather`.
4. Stable: renaming is breaking. Version by adding new names.
5. Namespace prefixes: `notes_list`, `notes_search`, `notes_create`.
6. No arguments in the name: `get_weather_for_city(city)`, not `get_weather_in_tokyo()`.

### Description pattern

```
Use when {condition}. Do not use for {close-but-wrong-cases}.
```

Stay under 1024 characters. Include format hints for parameters.

### Atomic vs monolithic

```python
# Monolithic (bad): forces the model to pick action from strings
do_everything(action: str, target: str, options: dict)

# Atomic (good): each has a tight description and typed schema
notes_list()
notes_create(title, body)
notes_delete(note_id)
```

Benchmarks show 15-30% worse selection on monolithic tools. Rule of thumb: if `action` has more than 3 values, split the tool.

### The linter

```python
def lint_name(name):
    findings = []
    if not SNAKE_CASE.match(name):
        findings.append(Finding("block", name, "name must be snake_case"))
    if any(m in name for m in TENSE_MARKERS):
        findings.append(Finding("warn", name, "name includes tense marker"))
    ...

def lint_description(desc, tool_name):
    findings = []
    if len(desc) < 40: findings.append(Finding("block", tool_name, "description under 40 chars"))
    if "use when" not in desc.lower(): findings.append(Finding("warn", tool_name, "missing 'Use when'"))
    ...
```

### Error messages as teaching signals

BAD: `TypeError: object of type 'NoneType' has no attribute 'lower'`
GOOD: `Invalid input: 'city' is required. Example: {"city": "Bengaluru"}.`

Typed error messages cut retry counts in half on weak models.

### Tool poisoning prevention

Descriptions land in the model's context verbatim. The linter rejects descriptions containing indirect-injection keywords: `<SYSTEM>`, `ignore previous`, URL shorteners.

## Use It

`code/main.py` ships a tool-schema linter that audits a registry against the design rules. Run on `GOOD_REGISTRY` (passes) and `BAD_REGISTRY` (fails on every rule).

## Exercises

1. Rewrite `BAD_REGISTRY` to pass the linter.
2. Design an MCP server for a notes app with atomic tools; lint to zero findings.
3. Pick a popular MCP server and find two actionable improvements in its tool descriptions.
4. Add the linter to your CI, failing on severity `block`.
5. Read Composio's field guide and add one uncovered rule to the linter.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| Tool description | Natural-language brief the model reads during selection |
| Atomic tool | One tool one action; name uniquely identifies behavior |
| Monolithic tool | Single tool with `action` string; selection accuracy tanks |
| Enum-closed set | `{type: "string", enum: [...]}` for closed domains |
| Tool poisoning | Hidden instructions in tool descriptions that hijack the agent |
| Tool-selection accuracy | Percentage of queries where the model picks correctly |
| Namespace prefix | `notes_*` — shared prefix grouping related tools |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/05-tool-schema-design)
