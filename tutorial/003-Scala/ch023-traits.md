# Traits

Traits are like Java interfaces — they define behaviour and are used for multiple inheritance to build loosely-coupled applications.

## Defining and Using Traits

```scala
abstract class Bike(val model: String) {
    // Implemented method
    def fuelCapacity = 30.0

    // Un-implemented (abstract) method
    def gearType: Double
}

trait Speed {
    def maxSpeed: Double
}

trait Engine {
    def engineType: String
    def engineOil: String
}
```

A class can extend one abstract class (or regular class) and mix in multiple traits using `extends` and `with`:

```scala
class Pulsar(model: String) extends Bike(model) with Speed with Engine {
    override def gearType: Double = 5.0
    override def fuelCapacity: Double = 35.0
    override def maxSpeed: Double = 300.00
    override def engineOil: String = "Castrol"
    override def engineType: String = "4 Stroke"
    override def toString(): String = s"Pulsar($model)"
}

val xp = new Pulsar("XP")
println(xp.maxSpeed)   // 300.0
println(xp.gearType)   // 5.0
println(xp.fuelCapacity)
```

## Anonymous Trait Instantiation

Traits cannot be instantiated directly, but you can create anonymous implementations:

```scala
val speedObj = new Speed {
    override def maxSpeed: Double = 250
}
println(speedObj.maxSpeed)
```

`speedObj` is not an instance of the `Speed` trait directly — it is an object of an anonymous class (`$$anon$$`) that implements `Speed`.

## Abstract Class vs Trait

| Feature | Abstract Class | Trait |
|---------|---------------|-------|
| Constructor parameters | Yes | No |
| Implemented methods | Yes | Yes |
| Multiple inheritance | No (single) | Yes (mix in many) |

Abstract classes can have constructor parameters; traits cannot. Both can have implemented and abstract methods. But a class can only extend one abstract class, while it can mix in many traits. This makes traits more flexible for composing behaviour.
