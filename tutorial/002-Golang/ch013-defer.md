# Defer

`defer` postpones execution of a function call until the surrounding function returns. It is primarily used for cleanup — closing files, releasing resources, unlocking mutexes — ensuring you never forget to clean up even if the function exits early.

```go
func main() {
    fmt.Println("Resource Opened")
    defer fmt.Println("Resource Closed")
    fmt.Println("Resource in use")
}
// Output:
// Resource Opened
// Resource in use
// Resource Closed
```

## Key Behaviors

### Arguments are evaluated immediately, not when the deferred call runs

When a `defer` statement executes, its arguments are evaluated right away and stored. Even if the variables change later, the deferred call uses the original values:

```go
a := "Start"
defer fmt.Println(a)  // records "Start" now
a = "End"
// Output when function returns: "Start"
```

### Deferred calls are stacked (LIFO)

Multiple deferred calls are pushed onto a stack. When the function returns, they execute in last-in-first-out order:

```go
defer fmt.Println("first")   // pushed first, executed third
defer fmt.Println("second")  // pushed second, executed second
defer fmt.Println("third")   // pushed third, executed first
// Output:
// third
// second
// first
```

### Common Pattern: Resource Cleanup

The idiomatic way to handle resources is to defer the close immediately after a successful open:

```go
file, err := os.Open("file.txt")
if err != nil { return }
defer file.Close()
// ... work with file ...
// file.Close() runs automatically when this function returns
```

This pattern ensures cleanup happens regardless of where the function exits — whether normally, via an error return, or even a panic.
