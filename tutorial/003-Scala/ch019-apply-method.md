# The `apply` Method

In Scala, if a class or object defines a method named `apply`, you don't have to call it explicitly — simply passing arguments to the instance invokes `apply` automatically.

## `apply` in a Class

```scala
class Foo1(x: Int) {
    def apply(y: Int) = x + y
}

object MagicApply extends App {
    val foo1 = new Foo1(10)

    println(foo1.apply(3)) // 13 — explicit call
    println(foo1(3))       // 13 — implicit call, same as foo1.apply(3)
}
```

Writing `foo1(3)` is syntactic sugar for `foo1.apply(3)`. The compiler rewrites the call automatically.

## `apply` in an Object

Objects can also define `apply`:

```scala
object Foo2 {
    def apply(y: Int) = y + 5
}

println(Foo2(10)) // 15 — implicitly calls Foo2.apply(10)
```

This is why you can create case class instances without `new` — the compiler generates an `apply` method in the companion object that calls the constructor:

```scala
case class Foo3(x: Int)

val foo31 = Foo3.apply(10) // explicit
val foo32 = Foo3(12)       // implicit — same as Foo3.apply(12)

println(foo31) // Foo3(10)
println(foo32) // Foo3(12)
```

Case classes automatically get a companion object with an `apply` method that matches the constructor. `Foo3(12)` is just `Foo3.apply(12)`, which returns a new `Foo3` instance. This makes object creation concise and removes the need for the `new` keyword.
