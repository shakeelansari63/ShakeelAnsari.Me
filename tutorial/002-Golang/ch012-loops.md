# Loops

Go has a single looping construct: `for`. There is no `while` or `do-while`. The `for` keyword takes four different forms that cover all iteration needs.

## Four Forms

### 1. Classic three-part loop

```go
for i := 1; i <= 5; i++ {
    fmt.Println(i)
}
```

### 2. Condition-only loop (like `while`)

```go
j := 0
for j < 5 {
    fmt.Println(j)
    j++
}
```

### 3. Infinite loop (use `break` to exit)

```go
k := 0
for {
    k += 11
    if k > 50 {
        break
    }
}
```

### 4. Range loop (iterate over collections)

`range` returns the index/key and value for each element:

```go
lst := []int{1, 3, 5, 7, 9}
for key, val := range lst {
    fmt.Println("Key:", key, "value:", val)
}

mp := map[string]int{"k1": 1, "k2": 2}
for key, val := range mp {
    fmt.Println("Key:", key, "value:", val)
}
```

## Multi-Variable Loop

Go supports initializing and updating multiple variables in a single `for` statement. This is useful for sequences where the iteration state involves more than one value:

```go
for i, j := 1, 1; i <= 6 && j <= 10; i, j = i+1, j*i {
    fmt.Printf("i=%d j=%d\n", i, j)
}
```

Note the update expression `i, j = i+1, j*i` — Go's ability to assign multiple values simultaneously makes complex iteration patterns concise.
