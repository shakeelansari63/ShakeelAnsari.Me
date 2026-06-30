# For Comprehensions & FlatMap

## For Comprehension

For comprehensions are syntactic sugar for chains of `map`, `flatMap`, and `filter` operations. They make complex transformations more readable.

### Simple Map

```scala
val numbers = List(1, 2, 3, 4, 5)

// Using map
println(numbers.map(n => n * n))

// Using for comprehension — exactly equivalent
val squared = for (n <- numbers) yield n * n
println(squared)  // List(1, 4, 9, 16, 25)
```

### With Filter

```scala
// Using filter then map
println(numbers.filter(_ % 2 == 0).map(n => n * n))

// Using for comprehension with guard
val evenSquares = for {
    n <- numbers
    if n % 2 == 0
} yield n * n
println(evenSquares)  // List(4, 16)
```

### Nested Iteration (Cross Join)

For comprehensions handle nested `flatMap` + `map` elegantly:

```scala
val a = 1 to 5
val b = 4 to 9

// Using flatMap and map — cross join
println(a.flatMap(i => b.map(j => (i, j))))

// Same with for comprehension
println(for (i <- a; j <- b) yield (i, j))

// With filters on both collections
println(for {
    i <- a if (i % 2 == 1)
    j <- b if (j >= 7)
} yield (i, j))
```

## FlatMap

`flatMap` is like `map` but flattens nested collections by one level. Its signature is `flatMap(f: A => IterableOnce[B]): List[B]` (Scala 2.13+).

### Flattening Multi-dimensional Lists

```scala
val list = List(
    List(List(1, 2), List(3, 4)),
    List(List(5, 6), List(7, 8))
)

// flatMap reduces one dimension level
println(list.flatMap(x => x))          // 2D list
println(list.flatMap(x => x).flatMap(x => x))  // 1D list

// flatten does the same thing
println(list.flatten.flatten)
```

### FlatMap on 1D Lists

When the function returns a collection, `flatMap` combines `map` and `flatten`:

```scala
println(1 to 10 flatMap (x => List(-x, 0, x)))
// For each x, produces [-x, 0, x], then flattens
```

### FlatMap on Options

Since `Option` is a collection (0 or 1 elements), `flatMap` on a list of options filters out `None`:

```scala
println(
    List(Some(3), Some(5), None, None, Some(12), None).flatMap(x => x)
)
// List(3, 5, 12) — None values are removed, Some values unwrapped
```

### FlatMap on Maps

```scala
println(
    Map(1 -> "One", 2 -> "Two").flatMap(t =>
        Map(t._1 -> t._2, (t._1 * 100) -> (t._2 + " Hundred"))
    )
)
```

## Fold & Reduce

`fold` and `reduce` collapse a collection into a single value by repeatedly applying a binary operation.

```scala
val nums = List(1, 2, 3, 4, 5)
```

### Reduce

`reduce` takes the first element as the initial value and processes the rest:

```scala
val sum = nums.reduce(_ + _)
println(sum)  // 15
// Runs 4 times (5 elements, first is seed)
```

### Fold

`fold` takes an explicit seed value:

```scala
val product = nums.fold(1)(_ * _)
println(product)  // 120
// Runs 5 times (seed + 5 elements)
```

Fold can also return a different type:

```scala
println((1 to 10).foldLeft("0")((str, element) => str + ", " + element))
// "0, 1, 2, 3, ..., 10" — like mkString with initial value
```

### Left vs Right

- `foldLeft`/`reduceLeft` iterate left to right.
- `foldRight`/`reduceRight` iterate right to left, with parameters reversed.

```scala
println((1 to 10).foldLeft(0)(_ + _))   // left to right
println((1 to 10).foldRight(0)(_ + _))  // right to left
// Both give 55, but order of evaluation differs
```
