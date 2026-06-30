# Method Overriding

Method overriding allows a subclass to provide a different implementation for a method defined in its parent class. The overriding method must have the same signature as the parent class method.

## Overriding a Method

```scala
class Employee8(val firstName: String, val lastName: String) {
    require(firstName.nonEmpty, "First Name cannot be empty")
    require(lastName.nonEmpty, "Last Name cannot be empty")

    def fullName = s"$firstName $lastName"
}

class Manager8(firstName: String, lastName: String, val department: String) extends Employee8(firstName, lastName) {
    override def fullName = s"$firstName $lastName from $department Department"
}

val jd = new Employee8("John", "Doe")
val ac = new Manager8("Alan", "Croft", "Mathematics")

println(jd.fullName) // John Doe
println(ac.fullName) // Alan Croft from Mathematics Department
```

The `override` keyword is required in Scala. Without it, the compiler will give an error. This is intentional — it prevents accidental overriding when you add a method that happens to have the same name as a parent method.

## Polymorphism

The overridden method is used even when the object reference is typed as the parent class:

```scala
val acEmp: Employee8 = ac
println(acEmp.fullName) // Alan Croft from Mathematics Department
```

At runtime, Scala calls the actual object's method, not the reference type's method. This is **dynamic dispatch** or polymorphism.

## Overriding `equals`, `hashCode`, and `toString`

These three methods come from `AnyRef` (the root of the reference type hierarchy). Their default implementations compare references, not values.

### Default Behavior (without override)

```scala
val emp1 = new Employee8("John", "Doe")
val emp2 = new Employee8("John", "Doe")

println(emp1 == emp2)                 // false (reference comparison)
println(emp1.hashCode == emp2.hashCode) // false (different hash codes)
println(emp1)                         // Employee8@<some hex>  (unreadable)
```

### Overridden Behavior

```scala
class Employee9(val firstName: String, val lastName: String) {
    require(firstName.nonEmpty, "First Name cannot be empty")
    require(lastName.nonEmpty, "Last Name cannot be empty")

    def fullName = s"$firstName $lastName"

    override def equals(x: Any): Boolean = if (!x.isInstanceOf[Employee9]) false
    else {
        val otherEmp = x.asInstanceOf[Employee9]
        this.firstName == otherEmp.firstName && this.lastName == otherEmp.lastName
    }

    override def hashCode(): Int = {
        var result = 19
        result = 31 * result + firstName.hashCode
        result = 31 * result + lastName.hashCode
        result
    }

    override def toString(): String =
        s"Employee(FirstName = $firstName, LastName = $lastName)"
}

val emp3 = new Employee9("John", "Doe")
val emp4 = new Employee9("John", "Doe")

println(emp3 == emp4)                   // true (value comparison)
println(emp3.hashCode == emp4.hashCode) // true (same hash from same fields)
println(emp3)                           // Employee(FirstName = John, LastName = Doe)
```

Key rules when overriding these methods:
- **`equals`**: Must check that the other object is the same type before casting and comparing fields.
- **`hashCode`**: Must use the same fields as `equals`. Equal objects must have equal hash codes. The standard approach uses prime numbers (19, 31) for multiplication.
- **`toString`**: Return a human-readable representation of the object.

Note that case classes automatically generate sensible `equals`, `hashCode`, and `toString` implementations — you rarely need to write them by hand.
