# Lazy Val

A `lazy val` is not evaluated when it is declared. Instead, it is evaluated only on the first access. Subsequent accesses return the cached value without re-evaluating.

```scala
lazy val a = {
    println("Evaluated")
    5
}

println("Still no evaluation...")
println(a)  // prints "Evaluated" then 5
println(a)  // just prints 5 (cached, no re-evaluation)
```

When the line `lazy val a = ...` executes, the block is not evaluated. The string "Evaluated" is not printed until `a` is first accessed. On the second access, the cached value `5` is returned immediately.

## Use Case: Forward References

Lazy evaluation is useful when you need to use a value before it is declared. Normally, this won't compile:

```scala
// This won't compile:
// val b = c + 10   // c not initialized yet
// val c = 5
```

But with `lazy val`, the declaration order doesn't matter because neither is evaluated until first access:

```scala
lazy val b = c + 10
lazy val c = 5
println(b)  // 15
```

This works because when `b` is accessed, it evaluates `c + 10`, which triggers the evaluation of `c = 5`, producing `15`.

## Limitation

Only `val` can be lazy. `lazy var` is not allowed — the concept of lazy evaluation is tied to immutability.
