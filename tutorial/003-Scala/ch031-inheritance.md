# Inheritance (Subclasses)

Scala classes can inherit from other classes using the `extends` keyword. Subclasses are appropriate when there is an **Is-A** relationship between types.

## Basic Inheritance

```scala
class Employee7(val firstName: String, val lastName: String) {
    require(firstName.nonEmpty, "First Name cannot be empty")
    require(lastName.nonEmpty, "Last Name cannot be empty")

    def fullName = s"$firstName $lastName"
}

class Manager1(firstName: String, lastName: String, val department: String) extends Employee7(firstName, lastName)
```

`Manager1` extends `Employee7`, inheriting all its methods and fields. Notice that `firstName` and `lastName` are **not** declared with `val` or `var` in `Manager1` — they are simply constructor parameters passed to the parent class. The accessors are already defined in `Employee7`.

## Using Inherited Members

```scala
val jd = new Employee7("John", "Doe")
println(jd.fullName) // John Doe

val ac = new Manager1("Alan", "Croft", "Mathematics")
println(ac.fullName)      // John Doe (inherited from Employee7)
println(ac.department)    // Mathematics (only available on Manager1)
```

Both `Employee7` and `Manager1` objects can call `fullName`, but only `Manager1` has access to `department`.

## Type Hierarchy Checks

```scala
println(jd.isInstanceOf[Employee7]) // true — jd is an Employee7
println(ac.isInstanceOf[Manager1])  // true — ac is a Manager1
println(ac.isInstanceOf[Employee7]) // true — ac is also an Employee7 (parent class)
```

A child class instance is also an instance of its parent class. This follows the standard object-oriented principle of polymorphism.
