# Goroutines

Goroutines are lightweight threads managed by the Go runtime. The key difference from OS threads: OS threads have significant system overhead (each has its own memory and scheduler), while goroutines are **green threads** — an abstraction over OS threads. The Go runtime multiplexes many goroutines onto a smaller number of OS threads, making them cheap to create and manage.

```go
go sayHello()  // runs sayHello in a new goroutine

// Anonymous goroutine
go func() {
    fmt.Println("Hello Anonymous")
}()
```

## Main Completes Before Goroutines

The most common gotcha: a goroutine starts but the `main` function may finish before it runs, causing the program to exit:

```go
go sayHello()
// Program likely exits before sayHello executes
// Fix: time.Sleep or WaitGroup
time.Sleep(100 * time.Millisecond)
```

## Closure Capture Pitfall

Goroutines capture variables from the enclosing scope **by reference**. If the variable changes before the goroutine runs, it sees the new value:

```go
msg := "Hello"
go func() {
    fmt.Println(msg)  // might print "Updated message"
}()
msg = "Updated message"
```

To capture the value at the time the goroutine is created, pass it as a parameter:

```go
msg := "Hello"
go func(m string) {
    fmt.Println(m)  // prints "Hello" — the value at creation time
}(msg)
msg = "Updated message"
```

## WaitGroups

`time.Sleep` is not a reliable synchronization mechanism. Use `sync.WaitGroup` to wait for goroutines to finish:

```go
var wg sync.WaitGroup

wg.Add(1)  // increment counter — "I'm waiting for 1 more goroutine"
go func() {
    defer wg.Done()  // decrement counter when goroutine finishes
    // do work
}()
wg.Wait()  // blocks until counter reaches 0
```

## Mutexes

When multiple goroutines access shared data concurrently without synchronization, race conditions occur. A `sync.RWMutex` ensures only one goroutine can modify data at a time:

```go
var mt sync.RWMutex
var counter int

func increment() {
    mt.Lock()      // acquire exclusive lock
    counter++      // safe to modify
    mt.Unlock()    // release lock
}
```

Without the mutex, goroutines interleaving their reads and writes produce unpredictable results. With the mutex, they queue up and access the shared resource one at a time.

## GOMAXPROCS

Controls how many OS threads execute goroutines simultaneously. The default is the number of CPU cores:

```go
runtime.GOMAXPROCS(20)                     // set to 20 threads
fmt.Println(runtime.GOMAXPROCS(-1))        // query current setting
```
