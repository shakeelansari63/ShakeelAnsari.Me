# Advanced Types

## Algebraic Data Types (ADT)

ADTs model data using two kinds of composition:

- **Sum types** (OR relationship) — a value is one of several variants. Represented by `sealed trait` with `case class`/`case object` implementations.
- **Product types** (AND relationship) — a value is a combination of fields. Represented by `case class` parameters.

### Sum Types

```scala
sealed trait Shape
case class Circle(radius: Double) extends Shape
case class Rectangle(width: Double, height: Double) extends Shape

def area(shape: Shape): Double = shape match {
    case Circle(r)        => math.Pi * r * r
    case Rectangle(w, h)  => w * h
}
```

A `sealed` trait can only be extended within the same file. This enables exhaustive pattern matching — the compiler knows all possible subtypes:

```scala
sealed trait Weather
case object Sunny extends Weather
case object Cloudy extends Weather
case object Rainy extends Weather
case object Windy extends Weather

// Weather is a Sum Type — it is Sunny or Cloudy or Rainy or Windy
// (is-a relationship)
```

### Product Types

```scala
case class WeatherRequest(lat: Double, lon: Double)
// WeatherRequest has a lat and a lon (has-a relationship)
// It is the Cartesian product of lat and lon
```

### Hybrid Types (Sum of Products)

```scala
sealed trait WeatherResponse
case class GoodWeather(weather: Weather) extends WeatherResponse
case class BadWeather(weather: Weather) extends WeatherResponse
case class InvalidWeather(error: String, message: String) extends WeatherResponse
```

`WeatherResponse` is a Sum Type (it is Good, Bad, or Invalid), but each variant is a Product Type (has one or more fields).

## Self Types

Self types declare that a trait requires another type to be mixed in. This is a dependency declaration without inheritance:

```scala
trait Database {
    def query(sql: String): Unit
}

trait Service {
    self: Database =>  // Service requires Database to be mixed in
    def getUsers(): Unit = query("SELECT * FROM users")
}

class App extends Database with Service  // must mix in Database
```

Without `self: Database =>`, `Service` could not call `query`. The self type ensures a `Database` will be available.

### Why Not Inheritance?

If `Service extends Database`, that implies `Service` is a `Database` — but a service is not a database (wrong "is-a" relationship). Self types allow composition without implying inheritance:

```scala
trait Person {
    def hasAllergies(thing: Edible): Boolean
}

trait Diet { self: Person =>
    def canEat(thing: Edible): Boolean =
        if (hasAllergies(thing)) false else true
}

case object VegetarianAthlete extends Diet with Person {
    override def hasAllergies(thing: Edible): Boolean = thing match {
        case Fish | Chicken  => true
        case Spinach | Bread => false
    }
}
```

A `Diet` is not a `Person`, but it requires a `Person`. Self types express this relationship cleanly.

## Variance

Variance describes how generic types relate when their type parameters have subtype relationships:

```scala
class Animal
class Dog extends Animal
class Cat extends Animal
```

### Covariant `[+T]`

`Container[Parent]` is a subtype of `Container[Child]`:

```scala
class SomeList[+T]  // covariant
val a: SomeList[Animal] = new SomeList[Dog]  // works
```

### Invariant `[T]`

No subtyping relationship exists:

```scala
class SomePet[T]  // invariant (default)
// val b: SomePet[Animal] = new SomePet[Dog]  // error
```

### Contravariant `[-T]`

`Container[Child]` is a subtype of `Container[Parent]`:

```scala
class Vet[-T]  // contravariant
val c: Vet[Dog] = new Vet[Animal]  // works
```

### Variance Position Rules

- `val` fields are in **covariant position** — cannot use `-T` in a `val` of a contravariant class.
- `var` fields are in **invariant position** — only work with invariant types.
- Method parameters are in **contravariant position** — cannot use `+T` for a method parameter in a covariant class.
- Method return types are in **covariant position** — cannot use `-T` for a method return type in a contravariant class.

## Type Classes

A type class is a pattern using traits and implicits to add behaviour to existing types without modifying them:

```scala
trait Show[T] {
    def show(value: T): String
}

object Show {
    implicit val intShow: Show[Int] = (value: Int) => value.toString
    implicit val stringShow: Show[String] = (value: String) => value
}

def printValue[T](value: T)(implicit show: Show[T]): Unit = {
    println(show.show(value))
}

printValue(42)        // 42
printValue("Hello")   // Hello
// printValue(true)   // Error — no implicit Show[Boolean]
```

The `implicit` parameter `Show[T]` acts as a constraint: `printValue` can only be called with types that have an implicit `Show` instance in scope. This provides **ad-hoc polymorphism** — the behaviour is defined externally from the types themselves.
