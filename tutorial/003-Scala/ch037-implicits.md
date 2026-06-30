# Implicits

Implicits allow the Scala compiler to automatically fill in values or convert types. They are a core mechanism for type classes and ad-hoc polymorphism.

## Implicit Conversion

You can define an implicit conversion that automatically converts one type to another:

## Scala 2 Implicit Conversions

```scala
case class Person(name: String) {
    def greet = println(s"Hello, my name is $name, Lets Rock")
}

// Requires scala.language.implicitConversions in Scala 2.13+
implicit val stringToPerson: (String) => Person = (name: String) => Person(name)

// Scala automatically finds the implicit conversion from String to Person
"Alice".greet
```

When you call `"Alice".greet`, Scala sees that `String` has no `greet` method, finds the implicit conversion `stringToPerson`, and converts the string to a `Person` before calling `greet`.

## Scala 3 `given` / `using`

Scala 3 replaces implicits with `given` instances and `using` clauses:

```scala
case class Person(name: String):
    def greet = println(s"Hello, my name is $name, Lets Rock")

// Scala 3: given instance instead of implicit val
given Conversion[String, Person] = (name: String) => Person(name)

"Alice".greet
```

## Implicit Parameters (Scala 2) vs `using` (Scala 3)

### Scala 2
```scala
def greet(implicit name: String): String = s"Hello $name"

implicit val defaultName: String = "World"

println(greet)          // "Hello World" — uses implicit value
println(greet("Scala")) // "Hello Scala" — explicit value overrides
```

### Scala 3 equivalent
```scala
def greet(using name: String): String = s"Hello $name"

given defaultName: String = "World"

println(greet)            // "Hello World"
println(greet(using "Scala")) // "Hello Scala"
```

## Type Class Pattern

### Scala 2
```scala
implicit val multiplyBy2: List[Int] => Int = _.sum * 2
implicit val concatAndAppend2: List[String] => String = _.mkString + "2"

def doSomething[T](item: List[T])(implicit doable: List[T] => T) = doable(item)

println(doSomething(List(1, 2, 3, 4)))         // 20 (sum * 2)
println(doSomething(List("1", "2", "3", "4"))) // "12342" (concat + "2")
```

### Scala 3 equivalent
```scala
given multiplyBy2: Conversion[List[Int], Int] = _.sum * 2
given concatAndAppend2: Conversion[List[String], String] = _.mkString + "2"

def doSomething[T](item: List[T])(using doable: Conversion[List[T], T]) = doable(item)
```

The second parameter group tells the compiler to find a matching instance in scope. This is the foundation of the **type class pattern**, which provides ad-hoc polymorphism without modifying existing types.

> **Note:** This chapter shows Scala 2 syntax. If you are using Scala 3, prefer `given`/`using` over `implicit`.
