# Generics

Generics (added in Go 1.18) allow writing type-safe, reusable code without sacrificing performance. They avoid code duplication when the same logic applies to multiple types.

## The Problem

Without generics, you must write separate functions for each type:

```go
func AddInts(a, b int) int { return a + b }
func AddFloats(a, b float32) float32 { return a + b }
```

With types like `int8`, `int16`, `int32`, `int64`, `float32`, `float64`, this duplication is impractical.

## Type Parameters

A generic function uses **type parameters** in square brackets. The constraint lists which types are allowed:

```go
func Add[T int8 | int16 | int32 | int | float32 | float64](a T, b T) T {
    return a + b
}
```

## Constraint Interfaces

To avoid long inline type lists, define an interface that describes the permitted types:

```go
type Nums interface {
    int8 | int16 | int32 | int | float32 | float64
}

func AddNums[T Nums](a T, b T) T {
    return a + b
}
```

## Generic Structs

Structs can also have type parameters:

```go
type User[T Nums] struct {
    name string
    age  T
}

u1 := User[int]{name: "Jack", age: 20}
u2 := User[float32]{name: "John", age: 32.1}
```

The concrete type for `T` is specified at instantiation: `User[int]` or `User[float32]`.

## Generic Maps

You can define generic map types using the `comparable` constraint — an interface for types that support `==` and `!=`:

```go
type CustomMap[T comparable, V string | float32] map[T]V

m := make(CustomMap[int, string])
m[3] = "Hello"
```

The `comparable` constraint is required for map keys because map keys must be comparable. Built-in comparable types include all numeric types, strings, booleans, and pointers.
