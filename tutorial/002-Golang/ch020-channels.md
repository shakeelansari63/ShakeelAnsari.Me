# Channels

Channels are typed conduits for sending and receiving values between goroutines. They can only be created with `make` and are strongly typed — a `chan int` can only transmit `int` values.

## Unbuffered Channels

An unbuffered channel has no capacity. A **send blocks until a receiver is ready**, and a **receive blocks until a sender is ready**. This synchronizes the two goroutines.

```go
ch := make(chan int)

// Sender
go func() {
    data := 10
    ch <- data  // sends a copy — changing data afterwards won't affect receiver
    wg.Done()
}()

// Receiver
go func() {
    i := <-ch
    fmt.Println("Got:", i)
    wg.Done()
}()
```

## Directional Channels

By default, a channel is bidirectional. You can restrict direction in function parameters to make the API clearer and safer:

```go
// Receive-only — can only read from the channel
go func(ch <-chan int) {
    i := <-ch
    fmt.Println(i)
    wg.Done()
}(ch)

// Send-only — can only write to the channel
go func(ch chan<- int) {
    ch <- 42
    wg.Done()
}(ch)
```

## Buffered Channels

A buffered channel has capacity. The sender only blocks when the buffer is full; the receiver only blocks when the buffer is empty:

```go
ch2 := make(chan int, 10)  // buffer of 10
```

Without buffering, if a sender reaches `ch <- 42` with no receiver ready, it stops and waits. Similarly, a receiver at `<-ch` with no data waits. If both sides are waiting for each other, that is a **deadlock**.

## Range over Channel

The `range` keyword works with channels, yielding each value as it arrives. The loop continues until the channel is closed and drained:

```go
ch3 := make(chan int, 10)

go func(ch <-chan int) {
    for i := range ch {  // loops until close(ch)
        fmt.Println(i)
    }
    wg.Done()
}(ch3)

go func(ch chan<- int) {
    ch <- 10; ch <- 15; ch <- 12
    close(ch)  // must close to signal end of data
    wg.Done()
}(ch3)
```

Closing a channel is necessary to end a `range` loop. Without `close()`, the receiver blocks forever waiting for more data, causing a deadlock.

## Checking if a Channel is Closed

Reading from a closed channel returns the zero value immediately. To distinguish between a zero value and a closed channel, use the comma-ok form:

```go
i, isOpen := <-ch
fmt.Println(i, isOpen)  // isOpen=false means channel is closed
```

When `isOpen` is `false`, the channel has been closed and no more values will arrive.
