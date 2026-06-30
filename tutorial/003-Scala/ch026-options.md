# Options

`Option` represents an optional value — it can be either `Some(value)` or `None`. This is a safe alternative to using `null`.

```scala
var x: Option[String] = Some("Hello")
println(x)       // Some(Hello)
println(x.get)   // "Hello"

x = None
println(x)       // None
```

Once `x` is `None`, calling `.get` would throw a `NoSuchElementException`. Never use `.get` directly in production code.

## Safe Access

Use `.getOrElse` to provide a default value:

```scala
val y = Some("Hello")
val z = None

println(y.getOrElse("No Value"))  // "Hello"
println(z.getOrElse("No Value"))  // "No Value"
```

Or use pattern matching to handle both cases explicitly:

```scala
def getValue(x: Option[String]): String = x match {
    case Some(value) => value
    case None        => "No Value"
}
```

## Practical Example

Here is a class that uses `Option` for an optional field — the employee's middle name:

```scala
class Employee private (
    val firstName: String,
    val middleName: Option[String],
    val lastName: String
) {
    // Primary constructor is private — users go through these aux constructors
    def this(firstName: String, middleName: String, lastName: String) =
        this(firstName, Some(middleName), lastName)
    def this(firstName: String, lastName: String) =
        this(firstName, None, lastName)

    override def toString: String =
        s"Employee($firstName ${middleName.getOrElse("")} $lastName)"
}

val jd = new Employee("John", "Henry", "Doe")
val js = new Employee("Jon", "Snow")
println(jd)  // Employee(John Henry Doe)
println(js)  // Employee(Jon  Snow)
```

The private primary constructor forces users to go through the auxiliary constructors, which neatly wrap the optional middle name in `Some` or `None` as appropriate.
