# Collections

## Lists

Lists are immutable sequences. They are created with `List.apply` (the companion object's `apply` method), and support efficient head/tail decomposition:

```scala
val list = List(1, 2, 3, 4, 5)
println(list.head)   // 1 — first element
println(list.tail)   // List(2, 3, 4, 5) — everything except first
println(list(2))     // 3 — index access (zero-based)

// Lists can also be built using :: (cons) operator with Nil
val c = 1 :: 2 :: 3 :: 4 :: Nil
// :: is right-associative (ends in :), so this equals:
val d = Nil.::(4).::(3).::(2).::(1)

// List methods
println(list.max)       // 5
println(list.min)       // 1
println(list.isEmpty)   // false
println(list.mkString(", "))  // "1, 2, 3, 4, 5"
```

Lists have no restriction on duplicates and are zero-indexed.

## Arrays

Arrays are mutable collections, represented the same way as Java arrays under the JVM:

```scala
val arr = Array(1, 2, 3, 4, 5)
arr(0) = 10  // mutable — modifies in place
println(arr.mkString(", "))
// Arrays support most List methods
println(arr.max)    // 5
println(arr.head)   // 10
```

Arrays are often used with repeated (varargs) parameters:

```scala
def demoRepeatedParam(x: Int, y: String, z: Any*) = {
    println(x); println(y); println(z)
}

// Unravel a list into varargs with :_*
demoRepeatedParam(3, "Hi", List("yeah", "this", "is", "stupid", 3.00): _*)
```

## Sets

Sets are collections without duplicates. They are unordered and fast for membership tests:

```scala
val set = Set(1, 2, 3, 1, 2)  // duplicates removed
println(set)  // Set(1, 2, 3)

// Mathematical set operations
println(Set(1, 2, 3) union Set(3, 4, 5))    // Set(1,2,3,4,5)
println(Set(1, 2, 3) intersect Set(3, 4, 5)) // Set(3, 4)
println(Set(1, 2, 3) diff Set(3, 4, 5))      // Set(1, 2)

// Membership test
println(set(3))  // true — set.apply(3) checks membership
println(set(5))  // false
```

Since sets are unordered, there is no `head`, `tail`, or index-based access.

## Maps

Maps are key-value data structures. Keys must be unique — duplicate keys overwrite earlier entries:

```scala
val map = Map("a" -> 1, "b" -> 2, "c" -> 3)
// "a" -> 1 is syntactic sugar for ("a", 1), a Tuple2

println(map("a"))    // 1 — unsafe, throws if key missing
println(map.getOrElse("x", 0))  // 0 — safe with default

// Safe access returns Option
println(map.get("a"))  // Some(1)
println(map.get("x"))  // None
```

Symbols (interned strings) are a good fit for map keys:

```scala
val symMap = Map('SymbolA -> 1, 'SymbolB -> 2)
```

## Ranges

Ranges generate sequences of numbers without specifying them one by one:

```scala
val range1 = 1 to 10         // inclusive — includes 10
val range2 = 1 until 10      // exclusive — stops at 9
val range3 = 100 to 1 by -1  // descending

println(range1.mkString(", "))  // "1, 2, 3, ..., 10"
```

Ranges with step:

```scala
val byTwo = 1 until 10 by 2     // Range(1, 3, 5, 7, 9)
val down = 10 to 2 by -2        // Range(10, 8, 6, 4, 2)
```

## Common Collection Methods

```scala
val nums = List(1, 2, 3, 4, 5)

println(nums.mkString(", "))    // "1, 2, 3, 4, 5"
println(nums.sum)                // 15
println(nums.length)             // 5
println(nums.reverse)            // List(5, 4, 3, 2, 1)
```

### Useful Collection Methods

```scala
// Zip — interweave two collections (length = shortest)
println((1 to 5) zip (5 to 9))     // Vector((1,5), (2,6), ...)

// ZipWithIndex — add index to each element
println(List("A", "AB", "AC").zipWithIndex)

// Partition — split collection by predicate
println((1 to 10).partition(_ % 2 == 1))

// Take — take N elements from left/right
println((1 to 10).take(5))        // Range(1, 2, 3, 4, 5)
println((1 to 10).takeRight(5))   // Range(6, 7, 8, 9, 10)

// TakeWhile — take from left while predicate is true
println((1 to 10).takeWhile(_ <= 3))  // Range(1, 2, 3)

// GroupBy — create a map keyed by the grouping function
println(List("A", "AB", "AC", "BB", "BD", "DA").groupBy(_.head))

// Distinct — remove duplicates
println(List(1, 2, 3, 4, 2, 3, 5).distinct)  // List(1, 2, 3, 4, 5)
```
