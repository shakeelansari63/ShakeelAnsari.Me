# Constants

Constants are immutable values determined at compile time. They cannot be assigned a runtime value — for example, `math.Sin(10)` won't compile as a constant.

```go
const con1 int = 10
// const myCon float64 = math.Sin(10)  // runtime value — won't compile
```

## Iota

`iota` is a special symbol that acts as a counter. When assigned to constants in a `const` block, it starts at `0` and increments by `1` for each line. This is Go's mechanism for constant enumeration.

```go
const (
    a = iota  // 0
    b = iota  // 1
    c = iota  // 2
)
```

## Implicit Propagation

A key feature of `iota` is that it automatically propagates to subsequent lines. If you omit the `= iota` on later lines, the compiler assumes the same expression and increments the counter:

```go
const (
    k = iota  // 0
    l         // 1
    m         // 2
    n         // 3
)
```

## Skipping Zero

If you want to skip the zero value (for example, to start counting from 1), use the blank identifier `_`:

```go
const (
    _ = iota
    q         // 1
    w         // 2
    e         // 3
)
```

The write-only symbol `_` lets you discard the value of `iota` on that line, so the following constants start from 1.

## Iota Expressions

`iota` also supports expressions. The iota value is substituted into the expression and evaluated, and the expression propagates to subsequent lines just like the bare `iota`:

```go
const (
    ev1 = (iota + 1) * 2  // (0+1)*2 = 2
    ev2                    // (1+1)*2 = 4
    ev3                    // (2+1)*2 = 6
    ev4                    // (3+1)*2 = 8
)
```

This lets you generate sequences like even numbers, powers of two, or any pattern expressible as a function of the line index.
