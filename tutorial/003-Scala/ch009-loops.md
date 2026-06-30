# Loops

## While Loop

Here is a traditional while loop that prints values 100 to 1 in descending order, separated by commas:

```scala
var a = 100
var res = ""
while (a >= 1) {
    res = res + a
    if (a > 1) res = res + ", "
    a = a - 1
}
println(res)
```

## For Loop (For-Each)

Scala's for loops are actually for-each loops. They iterate over elements of a collection or range:

```scala
// Range with step — from 100 down to 1
for (a <- (100 to 1 by -1)) {
    // a goes from 100 down to 1
}

// Loop over a list to build squares
val items = List(1, 2, 3, 4, 5)
var squares = List[Int]()
for (l <- items) squares = squares :+ (l * l)
println(squares)
```

## For Comprehension (Yield)

Instead of mutating a variable inside a loop, Scala's **for comprehension** with `yield` generates a new collection. This is more functional and avoids mutation:

```scala
val numbers = List(1, 2, 3, 4, 5)
val squares = for (l <- numbers) yield (l * l)
println(squares)  // List(1, 4, 9, 16, 25)
```

## Functional Alternative

Many imperative loops can be replaced with purely functional collection operations. The while-loop example above can be rewritten without any mutable state:

```scala
val res2 = (1 to 100).reverse.mkString(", ")
println(res2)

// Or generate an already-reversed range
val res3 = (100 to 1 by -1).mkString(", ")
println(res3)
```

These functional alternatives are preferred in Scala — they are shorter, more readable, and eliminate mutable variables.
