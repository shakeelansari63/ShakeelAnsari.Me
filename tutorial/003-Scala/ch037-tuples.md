# Tuples

**Tuples** are immutable containers that can hold multiple values of possibly **different types**. They are useful for grouping related data without defining a dedicated class.

## Creating Tuples

```scala
val t = (1, "Hello", 43.20) // Tuple3[Int, String, Double]
println(t) // (1,Hello,43.2)
```

The type is inferred from the values. You can also specify the type explicitly:

```scala
// Using parentheses syntax
val t2: (String, Int, Boolean) = ("Hello", 32, false)
println(t2) // (Hello,32,false)

// Using TupleN class syntax
val t3: Tuple2[Int, Double] = (2, 22.4)
println(t3) // (2,22.4)
```

`(String, Int, Boolean)` is syntactic sugar for `Tuple3[String, Int, Boolean]`. Both are equivalent.

## Accessing Elements

Access tuple elements using `_1`, `_2`, `_3`, etc. Indices start at **1**, not 0:

```scala
println(t2._1) // Hello
println(t2._3) // false
println(t3._2) // 22.4
```

## Tuple Size Limit

The maximum tuple size is **Tuple22**. You can store up to 22 values in a tuple.

## Tuple2: swap

`Tuple2` has a special `swap` method that returns a new tuple with the two values exchanged:

```scala
val t4: Tuple2[Int, String] = (4, "T4")
val t5 = t4.swap
println(t4) // (4,T4)
println(t5) // (T4,4)
```

## The Arrow Syntax for Tuple2

The `->` operator creates a `Tuple2`:

```scala
val t6 = 2 -> "Two" // Creates Tuple2[Int, String]: (2, "Two")
println(t6) // (2,Two)
```

This is commonly used in Map literals: `Map(1 -> "one", 2 -> "two")`.

## Chaining Arrows

When you chain multiple `->` operators, each arrow creates a nested `Tuple2`:

```scala
val t7 = 7 -> "T7" -> "July" -> 7.00
println(t7) // (((7,T7),July),7.0)
```

The associativity of `->` means:
1. `7 -> "T7"` creates `(7, T7)`
2. `(7, T7) -> "July"` creates `((7, T7), July)`
3. `((7, T7), July) -> 7.00` creates `(((7, T7), July), 7.0)`

To access deeply nested values, chain the `_n` accessors:

```scala
println(t7._1._1._2) // "T7"
// t7._1       = ((7, T7), July)
// (t7._1)._1  = (7, T7)
// ((t7._1)._1)._2 = "T7"
```

Work through the nesting layer by layer: each `_1` peels off one level, then `_2` picks the second element of the result.
