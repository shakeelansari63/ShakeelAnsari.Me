# Infix Notation

In Scala, single-parameter methods can be called using **infix notation** — you can write `obj method arg` instead of `obj.method(arg)`. This works for any method with one parameter, including symbolic operators.

## Basic Infix Usage

```scala
class Foo4(x: Int) {
    def bar(y: Int) = x + y
    def baz(y: Int, z: Int) = x + y - z
    def qux(y: Int) = new Foo4(x + y)
    def *(y: Int) = x * y
    def apply(y: Int) = x - y
    def +:(y: Int) = x + y
}

object InfixOperator extends App {
    val foo = new Foo4(10)

    // Normal vs infix call
    println(foo.bar(5))  // 15 — normal
    println(foo bar 5)   // 15 — infix
}
```

The infix form reads like a natural language expression: `foo bar 5` reads as "foo bar 5".

## Multiple Parameters

For methods with multiple parameters, the arguments must be grouped in parentheses:

```scala
println(foo.baz(5, 2))    // 13 — normal
println(foo baz (5, 3))   // 12 — infix with tuple-like parentheses
```

## Chaining Infix Calls

Infix notation chains naturally when methods return instances of the same type:

```scala
println(foo qux 5 qux 3 qux 9 qux 7 bar 12) // 46
```

This chains `qux` (creates a new `Foo4`) four times, then calls `bar` on the result. Each intermediate `Foo4` accumulates the value.

## Symbolic Methods

Methods can have symbolic names like `*`, which can also be used as infix operators:

```scala
println(foo.*(55)) // 550 — normal
println(foo * 55)  // 550 — infix
```

This is how Scala supports operator overloading — `*` is just a method name.

## `apply` as Infix

The `apply` method also works with infix syntax, though the direct call is more common:

```scala
println(foo.apply(4)) // 6
println(foo apply 4)  // 6 — infix
println(foo(4))       // 6 — implicit apply
```

## Right-Associative Colon Methods

If a method name **ends with a colon** (`:`), it is **right-associative**. The receiver and argument swap positions:

```scala
println(foo.+:(55)) // 65 — normal: foo.+:(55)
println(55 +: foo)  // 65 — infix: foo.+:(55), but written with 55 on the left
```

When you write `55 +: foo`, the compiler rewrites it as `foo.+:(55)`. The method is called on the **right** operand, not the left. This is the same mechanism that powers `::` for list construction (`1 :: 2 :: Nil`).
