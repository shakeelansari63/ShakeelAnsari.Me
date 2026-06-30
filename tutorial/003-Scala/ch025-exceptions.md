# Exceptions

## Try-Catch-Finally

Scala uses `try-catch-finally` blocks for exception handling. Unlike Java, the `catch` block uses pattern matching with `case` statements instead of separate exception type declarations:

```scala
try {
    throw new IllegalArgumentException("Bad data")
} catch {
    case iae: IllegalArgumentException =>
        println(iae.getMessage)
} finally {
    println("Always runs")
}
```

The `finally` block always executes, whether an exception was thrown or not.

## Functional Error Handling with Try

Scala also provides a more functional approach using `Try`, `Success`, and `Failure` from `scala.util`. This avoids side effects and wraps success or failure in a type-safe container:

```scala
import scala.util.{Try, Success, Failure}

def mayThrowError: Try[String] = Try(throw new IllegalArgumentException("Bad"))
def maySucceed(x: String): Try[String] = Try("Hello " + x)

val result = maySucceed("Damon")
result match {
    case Success(value) => println(value)
    case Failure(ex)    => println(ex.getMessage)
}
```

With `Try`, you don't need to worry about exceptions propagating unexpectedly. The result is always a `Try` value — either `Success` containing the result or `Failure` containing the exception. You then use pattern matching to handle each case explicitly.

This approach is preferred in functional Scala because it makes error handling explicit and composable, rather than relying on side-effectful try-catch blocks.
