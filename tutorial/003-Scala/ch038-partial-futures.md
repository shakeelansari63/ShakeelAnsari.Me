# Partial Functions & Futures

## Partial Functions

A partial function is defined only for a subset of possible input values. It should not be confused with partially applied functions (which fix some arguments of a multi-parameter function).

```scala
// A total function — defined for all Int values
val square: Int => Int = (x: Int) => x * x
println(square(32))

// A partial function — only defined for even numbers
val divideByTwo: PartialFunction[Int, Int] = {
    case x if x % 2 == 0 => x / 2
}

println(divideByTwo(4))              // 2
println(divideByTwo.isDefinedAt(3))   // false
```

### Creating Partial Functions

```scala
val squarePartialFunc: PartialFunction[Int, Int] = {
    case x if x < 10 => x * x * x
    case x if x < 20 => x * x
}
```

Under the hood, partial functions implement the `PartialFunction` trait with `apply` and `isDefinedAt`:

```scala
val pf1 = new PartialFunction[Int, String] {
    override def apply(x: Int): String = x match {
        case 1 => "One"
        case 2 => "Two"
        case 3 => "Three"
    }
    override def isDefinedAt(x: Int): Boolean =
        x == 1 || x == 2 || x == 3
}
```

### Useful Methods

```scala
// isDefinedAt — check if function is defined for a value
println(squarePartialFunc.isDefinedAt(2))   // true
println(squarePartialFunc.isDefinedAt(22))  // false

// lift — convert to total function returning Option
val optionalSquare: Int => Option[Int] = squarePartialFunc.lift
println(optionalSquare(5))  // Some(125)
println(optionalSquare(25)) // None

// orElse — combine multiple partial functions
val extended = squarePartialFunc.orElse {
    case x if x < 30 => x
}
```

### Using Partial Functions with Collections

```scala
// With map — will throw MatchError for undefined values
val list = List(1, 2, 3) map {
    case 1 => 32
    case 2 => 42
    case 3 => 56
}

// With collect — filters and transforms in one step
val anotherList = List(1, 2, 3, 4, "Hello", "Junkie") collect {
    case x if x.isInstanceOf[Int] => x
}
println(anotherList)  // List(1, 2, 3, 4)
```

## Asynchronous Programming with Futures

`Future` represents a computation that may not have completed yet. It runs on a separate thread and allows non-blocking execution.

```scala
import scala.concurrent.{Future, Await}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._

// A blocking method
def blockingMethod(x: Int): Int = {
    println("Sleeping for 5 seconds...")
    Thread.sleep(5000)
    x + 42
}

// An asynchronous version using Future
def asyncMethod(x: Int): Future[Int] = Future {
    println("Sleeping for 5 seconds...")
    Thread.sleep(5000)
    x + 42
}

// The async method runs on its own thread
println(asyncMethod(22))
// Following statement executes immediately
println("Done Waiting...")
```

### Handling Results

`Future` results can be handled with `onComplete`:

```scala
asyncMethod(22).onComplete {
    case Success(value)     => println(value)
    case Failure(exception) => println(s"Error: ${exception.getMessage}")
}
```

### Blocking for Results

Use `Await` to block until a future completes (usually only in test code or at the edge of your program):

```scala
import scala.concurrent.duration._
val result = Await.result(asyncMethod(22), atMost = 10.seconds)
println(result)  // 64

// Future supports map and flatMap
val mapped = Await.result(
    asyncMethod(22).map(x => x + 10),
    atMost = 10.seconds
)
println(mapped)  // 74
```

### Working with Multiple Futures

```scala
val f1 = Future { Thread.sleep(500); "Future 1" }
val f2 = Future { Thread.sleep(500); "Future 2" }
val f3 = Future { Thread.sleep(500); "Future 3" }

// First completed future wins
println(Await.result(
    Future.firstCompletedOf(List(f1, f2, f3)),
    1.second
))

// Wait for all futures
println(Await.result(
    Future.sequence(List(f1, f2, f3)),
    1.second
))
```

## Monads & Functors

### Functor

A **Functor** is a type that implements `map` — it transforms the value(s) inside a context while preserving the structure:

```scala
List(1, 2, 3).map(_ * 2)   // List(2, 4, 6)
Some(5).map(_ + 1)          // Some(6)
```

### Monad

A **Monad** is a type that implements `flatMap` — it chains operations that themselves produce monadic values. This composition follows three laws:

**Left Identity**: `Monad(x).flatMap(f) == f(x)`
```scala
val twoConsecutive: Int => List[Int] = x => List(x, x + 1)
println(twoConsecutive(3))           // List(3, 4)
println(List(3).flatMap(twoConsecutive))  // List(3, 4)
```

**Right Identity**: `Monad(x).flatMap(y => Monad(y)) == Monad(x)`
```scala
println(List(1, 2, 3).flatMap(x => List(x)))  // List(1, 2, 3)
```

**Associativity**: `Monad(x).flatMap(f).flatMap(g) == Monad(x).flatMap(x => f(x).flatMap(g))`
```scala
val incrementor: Int => List[Int] = x => List(x, x + 1)
val doubler: Int => List[Int] = x => List(x, x * 2)

println(List(1, 2, 3).flatMap(incrementor).flatMap(doubler))
println(List(1, 2, 3).flatMap(x => incrementor(x).flatMap(doubler)))
// Both produce: List(1, 2, 2, 4, 2, 4, 3, 6, 3, 6, 4, 8)
```

For-comprehensions are syntactic sugar for `flatMap` and `map` chains:

```scala
val result = for {
    x <- Some(5)
    y <- Some(10)
} yield x + y
println(result)  // Some(15)
```

Common monads in Scala: `Option`, `Try`, `Future`, `List`, `Either`.
