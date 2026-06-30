# Units

`Unit` in Scala is like `void` in other languages — it represents the absence of a meaningful value. A `Unit` value is created using empty parentheses `()`.

```scala
val a = ()
println(a)           // prints ()
println(a.getClass)  // prints void (the Java void class)
```

`Unit` is a subclass of `AnyVal`, meaning it is considered a value type. This has implications for type inference when a method can return either `Unit` or a concrete value:

```scala
def add(a: Int, b: Int) = if (a > b) println(s"$a > $b")
                          else a + b

println(add(3, 4))  // prints 7 (Int)
println(add(4, 3))  // prints "4 > 3" then () (Unit)
```

Because this method can return either `Int` or `Unit`, the inferred return type becomes `AnyVal` (the common supertype of both).

## The `=` Sign Determines Return

The `=` sign in a method definition tells Scala to return the value of the expression. If you omit `=`, the method returns `Unit` regardless of the body:

```scala
def goodAdd(a: Int, b: Int) = { a + b }   // returns sum — 7
def badAdd(a: Int, b: Int) { a + b }      // returns Unit — ()

println(goodAdd(3, 4))  // 7
println(badAdd(3, 4))   // ()
```

The `badAdd` method without `=` is equivalent to explicitly declaring a `Unit` return type:

```scala
def badAdd2(a: Int, b: Int): Unit = { a + b }
println(badAdd2(3, 4))  // ()
```

Even though the body computes `a + b`, the result is discarded because the method is declared to return `Unit`.
