# Parallel Tool Calls and Streaming with Tools

> Three independent weather lookups serialized is three round trips. Run them in parallel and total time collapses to the slowest single call. Every frontier provider now emits multiple tool calls in a single turn. The payoff is real; the plumbing is subtle.

**Type:** Build
**Languages:** Python (stdlib, thread pool + streaming harness)
**Prerequisites:** Phase 13 · 02
**Time:** ~75 minutes

## Learning Objectives
- Explain why `parallel_tool_calls: true` exists and when to disable it
- Correlate streamed argument chunks to the right tool-call id during parallel fan-out
- Reassemble partial `arguments` strings into complete JSON without parsing early
- Run a three-city weather benchmark demonstrating sequential vs parallel latency

## The Problem

Without parallel calls, three weather lookups cost three LLM round trips plus executor latency. With parallel calls: one LLM round trip, executor time is max of three not sum. Production benchmarks show 60-70% wall-clock reduction.

## The Concept

### Enabling parallel

- **OpenAI**: `parallel_tool_calls: true` (default). `false` for serial.
- **Anthropic**: `disable_parallel_tool_use: false` (default on Claude 3.5+).
- **Gemini**: Always parallel-capable.

Disable when tools have ordering dependencies (`create_file` then `write_file`) or when rate limiters cannot handle fan-out.

### Id correlation

Every call has an `id`. Every result must echo the same id. Without it, results are ambiguous.

### Running calls concurrently

```python
def run_parallel(cities):
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(cities)) as pool:
        results = list(pool.map(executor_weather, cities))
    dt_ms = (time.perf_counter() - start) * 1000
    return dt_ms, results
```

### Streaming tool calls — the accumulator pattern

Arguments arrive in pieces. Three parallel call streams interleave on the wire. You need one accumulator per id.

```python
@dataclass
class CallBuffer:
    id: str; name: str = ""; args_buf: str = ""; done: bool = False

@dataclass
class StreamAccumulator:
    buffers: dict[str, CallBuffer] = field(default_factory=dict)
    def on_event(self, event):
        kind = event["type"]; idx = event.get("id")
        if kind == "call_start":
            self.buffers[idx] = CallBuffer(id=idx, name=event["name"])
        elif kind == "args_delta":
            self.buffers[idx].args_buf += event["chunk"]
        elif kind == "call_stop":
            self.buffers[idx].done = True
            completed.append(self.buffers[idx])
        return completed
```

### The parse-early trap

Partial JSON like `{"city": "Beng` is not valid. Wait for the provider's end-of-call signal (OpenAI's `finish_reason="tool_calls"`, Anthropic's `content_block_stop`) before parsing.

### Benchmark

Three executors with 400, 600, 800 ms latency: sequential = 1800 ms, parallel = 800 ms. The savings grow with tool count.

## Use It

`code/main.py` has two halves: sequential vs parallel weather calls with wall-clock timing, and a fake streaming response reassembly using `StreamAccumulator`.

```python
seq_ms, _ = run_sequential(cities)
par_ms, _ = run_parallel(cities)
print(f"speedup: {seq_ms/par_ms:.2f}x")
```

## Exercises

1. Vary simulated latencies and confirm the parallel-to-sequential ratio approximates `max/sum`.
2. Extend the accumulator to handle a "call was cancelled mid-stream" case.
3. Replace the thread pool with `asyncio.gather` and benchmark.
4. Add an `ordering_dependency` graph to gate parallel fan-out.
5. Identify the real-world tool type where Anthropic recommends disabling parallelism.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| Parallel tool calls | Multiple tool calls in one assistant message |
| `disable_parallel_tool_use` | Anthropic's opt-out flag |
| Accumulator | Per-id string buffer for partial arguments chunks |
| Out-of-order completion | Parallel calls finish in unpredictable order; ids are the glue |
| Dependency graph | Tools whose outputs feed into inputs of other tools |
| Parse-early trap | Attempting to parse an incomplete arguments string |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/03-parallel-and-streaming-tool-calls)
