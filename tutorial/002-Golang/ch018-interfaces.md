# Interfaces

Interfaces define **behaviour** — they are composed of method signatures. Unlike structs (which describe data), interfaces describe what a type can do.

Go interfaces are **implicitly satisfied**: a type implements an interface simply by having the required methods. There is no `implements` keyword.

```go
type Writer interface {
    Write([]byte) (int, error)
}

type ConsoleWriter struct{}

// ConsoleWriter implicitly satisfies Writer
func (cw ConsoleWriter) Write(data []byte) (int, error) {
    return fmt.Println(string(data))
}

var w Writer = ConsoleWriter{}
w.Write([]byte("Hello Console !!!"))
```

## Interface Composition

Interfaces can be composed from other interfaces, allowing you to build complex behaviour contracts from simple pieces:

```go
type Shape interface { Area() float64 }
type Measurable interface { Perimeter() float64 }

type Geometry interface {
    Shape
    Measurable
}

type Rectangle struct {
    width, height float64
}

func (r Rectangle) Area() float64      { return r.width * r.height }
func (r Rectangle) Perimeter() float64 { return 2*r.width + 2*r.height }

func describeGeometry(g Geometry) {
    fmt.Println("Area:", g.Area(), "Perimeter:", g.Perimeter())
}
```

Since `Rectangle` implements both `Shape` and `Measurable`, it automatically satisfies `Geometry` — no explicit declaration needed.

## Empty Interface / `any`

The empty interface `interface{}` (or `any` in Go 1.18+) has no methods, so **every type satisfies it**. It is Go's way of representing "any type":

```go
var i1 interface{} = 3  // or: var i1 any = 3
```

## Type Assertion

To extract a concrete value from an interface, use a **type assertion**. The comma-ok form prevents panics if the assertion fails:

```go
if i2, ok := i1.(int); ok {
    fmt.Println("Extracted:", i2)
}
```

Without the `ok` check, a failed assertion panics. Always use the comma-ok form when you are not certain of the type.

## Custom Errors

The `error` type in Go is itself an interface:

```go
type error interface {
    Error() string
}
```

Any type that implements `Error() string` satisfies the `error` interface, allowing you to create custom error types:

```go
type CalculationError struct{ msg string }
func (c CalculationError) Error() string { return c.msg }

func divideNumbers(num, den float64) (float64, error) {
    if den == 0 {
        return 0, CalculationError{"cannot divide by 0"}
    }
    return num / den, nil
}
```

This is the foundation of Go's error handling — `error` is just an interface, and any type can become an error by implementing one method.
