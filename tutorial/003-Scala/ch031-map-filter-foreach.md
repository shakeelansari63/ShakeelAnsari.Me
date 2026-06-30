# Map, Filter & ForEach

These are the three most fundamental functional operations on collections.

## ForEach

`foreach` applies a side-effecting function to each element. Unlike `map`, it returns `Unit` (nothing), not a new collection:

```scala
val list = List(1, 2, 3, 4, 5)
list.foreach(println)

list.foreach(x => println(x * 2))
```

`foreach` is defined as `foreach(f: A => Unit): Unit` — it is used purely for side effects like printing or writing to a database.

## Map

`map` transforms each element and returns a new collection. It is defined as `map(f: A => B): Collection[B]`:

```scala
val numbers = List(1, 2, 3, 4, 5)
val doubled = numbers.map(_ * 2)
println(doubled)  // List(2, 4, 6, 8, 10)

val words = List("hello", "world")
val uppercased = words.map(_.toUpperCase)
println(uppercased)  // List(HELLO, WORLD)
```

Map works on all collection types — Lists, Sets, Ranges, Strings (character by character), Maps, and even Options:

```scala
// Map on Set
println(Set("A", "AB", "ABC").map(_.size))  // Set(1, 2, 3)

// Map on Range
println(1 to 10 by 2 map (_ * 2))

// Map on String (works on each character)
println("Hello!!".map(c => (c + 2).toChar).mkString)

// Map on Map
val c = Map("FG" -> 1, "BG" -> 2)
println(c.map(t => "Key: " + t._1))

// Map on Option — transforms the inner value if present
println(Some(2).map(_ + 3))  // Some(5)
println(None.asInstanceOf[Option[Int]].map(_ + 3))  // None
```

## Filter

`filter` keeps elements that satisfy a predicate. It takes a function `A => Boolean`:

```scala
val numbers = List(1, 2, 3, 4, 5, 6)
val evens = numbers.filter(_ % 2 == 0)
println(evens)  // List(2, 4, 6)
```

Related methods:

```scala
// filterNot — inverse of filter
println(List(1, 2, 3, 4, 5, 6).filterNot(_ % 2 == 0))  // List(1, 3, 5)

// exists — check if any element satisfies predicate
println(List(1, 2, 3, 4, 5, 6).exists(_ > 10))  // false

// withFilter — lazy version of filter, more memory efficient
println((1 to 10).withFilter(_ % 3 == 0).map(i => i))
```

Filter works on all collection types — List, Range, Set, Map, and even Option:

```scala
// Filter on Set
println(Set("Red", "Green", "Indigo").filter(_ contains 'e'))

// Filter on Map
println(Map(1 -> "One", 2 -> "Two").filter(t => t._2 contains 'e'))

// Filter on Option — None if predicate fails, Some if passes
println(Some(5).filter(_ % 2 == 0))  // None (5 is not even)
println(Some(4).filter(_ % 2 == 0))  // Some(4)
```

## Chaining

You can chain these operations together for expressive data pipelines:

```scala
val result = (1 to 20)
    .filter(_ % 2 == 0)
    .map(x => x * x)
    .foreach(println)
// Prints squares of even numbers from 1 to 20
```
