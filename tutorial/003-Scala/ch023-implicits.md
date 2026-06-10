# Implicits

Implicits allow the Scala compiler to automatically fill in values or convert types. They are a core mechanism for type classes and ad-hoc polymorphism.

## Implicit Conversion

You can define an implicit conversion that automatically converts one type to another:

```scala
case class Person(name: String) {
    def greet = println(s"Hello, my name is $name, Lets Rock")
}

implicit val stringToPerson: (String) => Person = (name: String) => Person(name)

// Scala automatically finds the implicit conversion from String to Person
"Alice".greet
```

When you call `"Alice".greet`, Scala sees that `String` has no `greet` method, finds the implicit conversion `stringToPerson`, and converts the string to a `Person` before calling `greet`.

## Implicit Parameters

Implicits can be used as function parameters. The compiler automatically fills them in from the implicit scope:

```scala
def greet(implicit name: String): String = s"Hello $name"

implicit val defaultName: String = "World"

println(greet)          // "Hello World" — uses implicit value
println(greet("Scala")) // "Hello Scala" — explicit value overrides implicit
```

## Type Classes with Implicits

Implicits are commonly used with curried methods to implement the type class pattern:

```scala
implicit val multiplyBy2: List[Int] => Int = _.sum * 2
implicit val concatAndAppend2: List[String] => String = _.mkString + "2"

def doSomething[T](item: List[T])(implicit doable: List[T] => T) = doable(item)

println(doSomething(List(1, 2, 3, 4)))         // 20 (sum * 2)
println(doSomething(List("1", "2", "3", "4"))) // "12342" (concat + "2")
```

The second parameter group `(implicit doable: List[T] => T)` tells the compiler to find an implicit value of the appropriate type in scope. For `List[Int]`, it finds `multiplyBy2`; for `List[String]`, it finds `concatAndAppend2`. If no matching implicit is in scope, the code will not compile.

This is the foundation of the **type class pattern**, which provides ad-hoc polymorphism without modifying existing types.
