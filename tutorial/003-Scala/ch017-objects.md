# Objects

Scala does not have `static` classes or methods. Instead, `object` creates a **singleton** — a class with exactly one instance. Objects are used for namespacing utility functions, factory methods, and as companions.

## Basic Object

```scala
object MyObject {
    def sayHello = println("Hello")
}

MyObject.sayHello
println(MyObject)  // MyObject$@<hash>
```

## Object Inheritance

Objects cannot be inherited from, but they **can** inherit from other classes or traits:

```scala
case class Greeting(msg: String)

object MyGreeting extends Greeting("How are you??") {
    def greet = println(msg)
}

MyGreeting.greet
```

Since `toString` is defined in the case class, `println(MyGreeting)` prints `Greeting(How are you??)`.

## Case Objects

Just like case classes, **case objects** get automatic `toString`, `equals`, `hashCode`, and pattern matching support:

```scala
class Weather(val name: String)

// Regular object — toString not overridden
object Sunny extends Weather("Sunny")

// Case object — nice toString and other benefits
case object SunnyX extends Weather("Sunny")

println(Sunny)   // Weather@<hash>
println(SunnyX)  // SunnyX
```

Case objects are the idiomatic way to create singleton values, especially when used with pattern matching and sealed traits.
