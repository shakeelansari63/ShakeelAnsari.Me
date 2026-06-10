# Pattern Matching

Pattern matching in Scala is like `switch`/`case` in other languages, but much more powerful. You match a value against a set of patterns and execute the corresponding branch.

## Basic Matching

```scala
val num = 3
num match {
    case 1 => println("One")
    case 2 => println("Two")
    case 3 => println("Three")
    case _ => println("Default")
}
```

The wildcard `_` matches anything, similar to `default` in a switch statement.

## Match with Return Value

Pattern matching is an expression — it returns a value. This makes it useful inside functions:

```scala
def find(x: Int) = x match {
    case 1 => "One"
    case 2 => "Two"
    case 3 => "Three"
    case _ => "Default"
}

println(find(12))  // Default
```

## Pattern Matching with Guards

You can add conditional guards with `if` to make matches more specific:

```scala
val amount = 50
amount match {
    case x if x <= 50 => println(s"$x is <= 50")
    case x if x > 50  => println(s"$x is > 50")
}
```

Guards let you match ranges or complex conditions without listing every possible value.

## Matching on Types

You can match on the type of a value:

```scala
val something: Any = "Hello"
something match {
    case s: String => println(s"String: $s")
    case i: Int    => println(s"Int: $i")
    case _         => println("Other")
}
```

This is particularly useful when working with values of type `Any` or when you have a sealed class hierarchy.
