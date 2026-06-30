# No Value: Null, Nil, None, Nothing, Unit

Scala has several ways to represent the absence of a value. Each serves a different purpose and is suited to a different context.

## 1. `Null` / `null`

`null` is a **value** representing a missing reference for reference types. `Null` is the **type** of `null` — it can only hold the value `null`.

```scala
val a: String = null // No reference for a String variable
println(a) // null

val b: Null = null // Null type can only hold null
println(b) // null
```

`null` exists for Java interoperability, but idiomatic Scala avoids it. It leads to `NullPointerException` and there is no compile-time check to prevent it.

## 2. `Nil`

`Nil` is an **empty list** — a singleton representing `List[Nothing]`:

```scala
val c: List[Int] = Nil
println(c) // List()

val d = 1 :: 2 :: 3 :: Nil
println(d) // List(1, 2, 3)
```

`Nil` is the base case for list construction. Every non-empty list ends with `Nil`.

## 3. `None`

`None` is the empty value for the `Option` type. `Option` is a container that is either `Some(value)` or `None`:

```scala
val e: Option[Int] = None
println(e) // None

val f: Option[Int] = Some(43)
println(f) // Some(43)
```

`Option` forces you to handle the missing-value case at compile time, eliminating null-pointer exceptions. Use `Option` whenever a value may or may not be present.

## 4. `Unit`

`Unit` is equivalent to `void` in Java or C. It indicates that a function returns no meaningful value — it executes only for its side effects:

```scala
val g: () => Unit = () => println("Hello I return Unit")
val h = g() // prints "Hello I return Unit"
println(h) // ()
```

`h` is of type `Unit` and its value is `()`. Every expression in Scala must return something, so `Unit` is the "return nothing" value.

## 5. `Nothing`

`Nothing` is a subtype of **every** type in Scala. It has no instances — the only way to produce a `Nothing` value is to throw an exception:

```scala
val i: Int = throw new RuntimeException("Can't Work like this")
```

Because `Nothing` is a subtype of `Int`, the assignment to `val i: Int` compiles. The exception is thrown before the assignment completes. `Nothing` is useful for defining the return type of methods that never return normally (e.g., `error`, `assert` failures, infinite loops).

## Summary

| Concept | Kind | Use Case |
|---------|------|----------|
| `null` / `Null` | Value / Type | Java interop (avoid in Scala code) |
| `Nil` | Value | Empty list / list terminator |
| `None` | Value | Absent value in `Option` |
| `Unit` | Type / Value | Side-effect-only functions |
| `Nothing` | Type | Bottom type for exceptions / non-termination |
