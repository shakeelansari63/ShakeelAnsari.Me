# Context

Context provides a standard way to carry deadlines, cancellation signals, and request-scoped values across API boundaries and goroutines. It prevents hanging connections by letting you set a timeout for operations.

## Timeout Context

```go
ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
defer cancel()

req, _ := http.NewRequestWithContext(ctx, http.MethodGet,
    "https://example.com/data", http.NoBody)
resp, err := http.DefaultClient.Do(req)
if err != nil {
    panic(err)  // "context deadline exceeded" if too slow
}
defer resp.Body.Close()

data, _ := io.ReadAll(resp.Body)
fmt.Printf("Image size: %d\n", len(data))
```

## How It Works

`context.WithTimeout` creates a derived context that auto-cancels after the specified duration. It returns:
- `ctx` — the context with the timeout configured
- `cancel` — a function to explicitly cancel early (resources are freed when this is called)

`context.Background()` is Go's built-in root context — it is never cancelled and has no deadline. You always derive child contexts from it with `WithTimeout`, `WithCancel`, or `WithValue`.

The `cancel` function must be called (typically via `defer`) to release the context's resources. Without it, the timeout is the only way the context is cleaned up.

`http.NewRequestWithContext` returns a request that the HTTP client will cancel automatically if the context expires before the response arrives. If the server is too slow, instead of hanging forever, the request fails immediately with a "context deadline exceeded" error.

Without context:
- A slow HTTP call blocks the goroutine indefinitely
- No way to set timeouts without manual timer management

With context:
- Clear timeout control
- Automatic cancellation propagation
- Standard interface for any library to respect deadlines
