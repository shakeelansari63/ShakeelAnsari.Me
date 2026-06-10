# Testing & Reflection

The `reflect` package provides runtime type inspection. It is the foundation for many Go libraries — JSON encoding, validation frameworks, and ORMs all use reflection internally.

## Basic Type Inspection

```go
import (
    "fmt"
    "reflect"
)

func main() {
    i := 1
    t := reflect.TypeOf(i)
    fmt.Println(t) // int
}
```

`reflect.TypeOf` returns the concrete type of a value. For an `int` it returns `"int"`; for a struct it returns `"package.StructName"`.

## Why This Matters

Reflection is how Go libraries implement generic-looking behaviour without generics (pre-Go 1.18). For example:

- `json.Marshal` uses reflection to read struct fields and their JSON tags
- `fmt.Println` uses reflection to inspect values and choose string formatting
- GORM reads `gorm` struct tags via reflection to map structs to database columns

## Writing Tests with `testing`

Go's standard `testing` package is the primary way to write unit tests. Test files are named `*_test.go` and placed alongside the code they test:

```go
// math.go
func Add(a, b int) int {
    return a + b
}
```

```go
// math_test.go
package main

import "testing"

func TestAdd(t *testing.T) {
    result := Add(2, 3)
    expected := 5
    if result != expected {
        t.Errorf("Add(2, 3) = %d; want %d", result, expected)
    }
}
```

Run tests with:

```bash
go test ./...
```

## Table-Driven Tests

A common Go pattern is table-driven tests — define a slice of test cases and iterate over them:

```go
func TestAdd(t *testing.T) {
    tests := []struct {
        a, b, expected int
    }{
        {1, 2, 3},
        {0, 0, 0},
        {-1, 1, 0},
    }
    for _, tt := range tests {
        result := Add(tt.a, tt.b)
        if result != tt.expected {
            t.Errorf("Add(%d, %d) = %d; want %d", tt.a, tt.b, result, tt.expected)
        }
    }
}
```

Table-driven tests make it easy to add new cases and see exactly which input failed.

## Coverage

```bash
go test -coverprofile=coverage.out
go tool cover -html=coverage.out
```

The `-coverprofile` flag writes coverage data. The `cover` tool generates an HTML report showing which lines are tested.

Reflection enables metaprogramming in Go, but prefer type-safe code when possible — reflection is slower, harder to read, and errors appear at runtime instead of compile time.
