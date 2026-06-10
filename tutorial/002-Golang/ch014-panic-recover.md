# Panic & Recover

Go has no exceptions in the traditional sense. Instead it uses `panic` and `recover` for truly exceptional situations.

- **`panic()`** stops ordinary execution, runs any deferred functions in the current function, then unwinds the stack running deferred functions in each caller, and finally crashes the program.
- **`recover()`** regains control of a panicking goroutine — it is only useful inside a deferred function.

## Without Recover

```go
// panic("something went wrong")  // program crashes with stack trace
```

## With Recover

To handle a panic gracefully, use `recover` inside a deferred function:

```go
defer func() {
    if r := recover(); r != nil {
        log.Println("Error:", r)
    }
}()
panic("something went wrong")
// Code after panic still won't run in same function — panic stops execution immediately
```

When `recover()` returns a non-nil value, the panic is "caught" and the program continues from the function that deferred the recover.

## Recover in a Called Function

If panic happens in a called function and recover is placed there, the **caller continues running normally**:

```go
func main() {
    fmt.Println("Start")
    someApplicationCanPanic()
    fmt.Println("End")  // This runs because panic was recovered
}

func someApplicationCanPanic() {
    defer func() {
        if err := recover(); err != nil {
            log.Println("Error:", err)
        }
    }()
    panic("something went wrong")
    fmt.Println("This won't run")  // execution stops at panic
}
```

This is a critical distinction: `recover` stops the unwinding **in the function where the deferred function is defined**. If you recover in `someApplicationCanPanic`, `main` does not know a panic occurred and continues as normal.

But if you only defer and recover in `main` (not in the called function), even though the deferred function runs, `main` has already ended — anything after the call that panicked will not execute.
