# Case Classes

Case classes bring several automatic features that reduce boilerplate compared to regular classes.

## Benefits

1. **Auto-generated `equals`, `hashCode`, and `toString`**
2. **No `new` keyword** needed for instantiation
3. **Parameters are `val` by default** — no need to write `val` for each parameter
4. **Built-in pattern matching support**
5. **`copy` method** for creating modified copies

## Comparison: Regular Class vs Case Class

```scala
// Regular class — no automatic equals, hashCode, or toString
class Employee10(val firstName: String, val lastName: String) {
    require(firstName.nonEmpty, "First Name cannot be empty")
    require(lastName.nonEmpty, "Last Name cannot be empty")
    def fullName = s"$firstName $lastName"
}

// Case class — everything comes for free
case class Employee11(firstName: String, lastName: String) {
    require(firstName.nonEmpty, "First Name cannot be empty")
    require(lastName.nonEmpty, "Last Name cannot be empty")
    def fullName = s"$firstName $lastName"
}
```

Let's compare their behavior:

```scala
// Regular class instances
val emp1 = new Employee10("John", "Doe")
val emp2 = new Employee10("John", "Doe")

println(emp1 == emp2)                    // false — different references
println(emp1.hashCode == emp2.hashCode)  // false
println(emp1)                            // Employee10@<hash> — not human readable

// Case class instances — no "new" keyword needed
val emp3 = Employee11("John", "Doe")
val emp4 = Employee11("John", "Doe")

println(emp3 == emp4)                    // true — structural equality
println(emp3.hashCode == emp4.hashCode)  // true
println(emp3)                            // Employee11(John, Doe) — readable
```

## Copy Method

The `copy` method creates a new instance with some fields changed:

```scala
val emp5 = emp3.copy(lastName = "Dae")
println(emp5)  // Employee11(John, Dae)
```

## Limitations

Case classes cannot inherit from other case classes. This is because `copy` has default parameter values, and if a subclass tried to override `copy`, it would conflict with the parent's `copy` method (Scala does not allow overloading methods with default parameters).

Case classes can, however, extend regular classes or traits.
