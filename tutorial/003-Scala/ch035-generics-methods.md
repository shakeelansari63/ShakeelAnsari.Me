# Generic Methods (Parametrized Types)

Parametrized types — also known as **generics** — allow methods to work with different types while preserving type information.

## The Problem with `Any`

If a method can return different types based on input, you might be tempted to use `Any`:

```scala
def anyDecide(b: Boolean, x: Any, y: Any) = if (b) x else y

println(anyDecide(true, 3, 'c'))   // 3
println(anyDecide(false, "Hello", 9)) // Hello
```

This works at runtime, but the return type is `Any`. If you need to pass the result to a method that expects a specific type, it won't compile:

```scala
def printInt(n: Int) = println(n)

// printInt(anyDecide(true, 3, 4)) // Error! Return type is Any, not Int
```

Even though `3` is an `Int` at runtime, the compiler only knows it's `Any` and refuses to pass it to a method expecting `Int`.

## Generic Type Parameter

Scala solves this with a **generic type parameter**:

```scala
def genericDecide[T](b: Boolean, x: T, y: T): T = if (b) x else y
println(genericDecide(true, 3, 5)) // 3

// Now we can pass the result directly to printInt
printInt(genericDecide(true, 3, 4)) // 4
```

The `[T]` introduces a type parameter that is inferred from the arguments. When both `x` and `y` are `Int`, `T` is inferred as `Int`, and the return type is `Int`. This means the compiler preserves the full type information through the call chain.
