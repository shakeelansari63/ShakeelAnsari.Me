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

In Scala 2, case classes cannot inherit from other case classes — `copy` has default parameter values that would conflict. In Scala 3, this restriction is lifted and case classes can extend other case classes.

Case classes can, however, extend regular classes or traits.
