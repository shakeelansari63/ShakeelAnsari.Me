# Conditional Flow

## If-Else

You can write if-else statements as in any other programming language:

```scala
val a = 10
var res = ""

if (a < 10) res = "Less"
else if (a > 10) res = "Greater"
else res = "Equal"

println(res)
```

This approach uses `var` because we need to change the value of `res`.

## If Returns a Value

In Scala, `if` is an **expression** — it returns a value. This means you can assign the result of an if-else directly to a `val`, avoiding mutation:

```scala
val b = 10
val res2 = if (b < 10) "Less"
           else if (b > 10) "Greater"
           else "Equal"

println(res2)  // Equal
```

This expression-oriented style is idiomatic Scala. Instead of mutating a variable inside branches, each branch becomes a value that the whole if-expression evaluates to. This eliminates the need for `var` and makes the code more predictable.
