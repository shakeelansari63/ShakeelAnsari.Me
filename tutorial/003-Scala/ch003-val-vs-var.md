# Val vs Var

In Scala, `val` is used to declare an immutable reference (a constant), while `var` is used to declare a mutable variable. Once a `val` is assigned, it cannot be reassigned. This is a fundamental concept in Scala — preferring immutability leads to safer, more predictable code.

## Example

```scala
var a = 10
val b = 10

// This will work — var can be reassigned
a = 5

// This will error: reassignment to val
// b = 5
```

`b = 5` would produce a compile-time error because `b` is declared with `val`. The compiler prevents reassignment to a `val`, enforcing immutability at compile time.

## Best Practice

Prefer `val` by default. Only use `var` when you genuinely need mutation. This aligns with Scala's functional programming philosophy — immutable data eliminates entire categories of bugs related to unexpected state changes.
