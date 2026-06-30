# Classes

## Basic Class

Classes in Scala are created with the `class` keyword, and all constructor parameters are provided in the class definition itself, which creates the default constructor:

```scala
class Employee1(firstName: String, lastName: String)
```

This creates a class with two constructor parameters. However, `firstName` and `lastName` are not automatically accessible as fields — they are just constructor parameters. Compiling this with `scalac` and analyzing with `javap -p Employee1` shows:

```
public class Employee1 {
  public Employee1(java.lang.String, java.lang.String);
}
```

To make them accessible, prefix them with `val` (immutable accessor) or `var` (mutable accessor):

```scala
class Employee2(val firstName: String, var lastName: String)
```

Now `javap` shows:

```
public class Employee2 {
  private final java.lang.String firstName;
  private java.lang.String lastName;
  public java.lang.String firstName();
  public java.lang.String lastName();
  public void lastName_$eq(java.lang.String);
  public Employee2(java.lang.String, java.lang.String);
}
```

- `val` creates a `final` field with a getter method (`firstName()`). The value cannot be changed after construction.
- `var` creates a mutable field with both a getter (`lastName()`) and a setter (`lastName_$eq()`), which is called when you write `obj.lastName = "new value"`.
- The field and accessor method have the same name. This is the **Uniform Access Principle** — variables and parameterless methods are accessed with the same syntax.

## Bean Properties

If you need Java-style getters (`getFirstName()`) and setters (`setLastName()`), use the `@BeanProperty` annotation:

```scala
import scala.beans.BeanProperty

class Employee3(
    @BeanProperty val firstName: String,
    @BeanProperty var lastName: String
)
```

This generates both Scala-style accessors and Java-style getters/setters:

```
public class Employee3 {
  private final java.lang.String firstName;
  private java.lang.String lastName;
  public java.lang.String firstName();
  public java.lang.String lastName();
  public void lastName_$eq(java.lang.String);
  public java.lang.String getFirstName();   // Java getter
  public java.lang.String getLastName();    // Java getter
  public void setLastName(java.lang.String); // Java setter
  public Employee3(java.lang.String, java.lang.String);
}
```

## Constructors

### Auxiliary Constructors

Auxiliary constructors are defined with `this(...)`. The first statement in every auxiliary constructor must call another constructor (ultimately the primary constructor):

```scala
class Person(val name: String, val age: Int) {
    def this(name: String) = this(name, 0)
    def this() = this("")
}
```

Auxiliary constructors can chain to each other. The primary constructor is always the ultimate target:

```scala
class ClassWithAuxConstructors(val param1: String, val param2: String, val param3: Int) {
    def this(param1: String, param2: String) = this(param1, param2, 0)
    def this(param1: String) = {
        this(param1, "", 0)
        println("Constructor with 1 parameter called")
    }
    def this() = {
        this("")     // calls constructor with 1 parameter
        println("Constructor with no parameters called")
    }
}
```

### Default Values & Named Parameters

Constructor parameters can have default values, and callers can use named arguments:

```scala
class Config(val host: String = "localhost", val port: Int = 8080)

val c1 = new Config("example.com", 3000)
val c2 = new Config(port = 9000)   // uses default host
```

Named arguments allow you to skip parameters with defaults and specify only the ones you need.

## Preconditions

Preconditions validate constructor inputs using the `require` keyword. They throw `IllegalArgumentException` if the condition fails:

```scala
class Employee(val firstName: String, val lastName: String) {
    require(firstName.nonEmpty, "First name cannot be empty")
    require(lastName.nonEmpty, "Last name cannot be empty")

    def fullName = s"$firstName $lastName"
}

// This throws: java.lang.IllegalArgumentException: requirement failed: Last name cannot be empty
val jd = new Employee(firstName = "John", lastName = "")
```

Preconditions are checked during object creation, before the constructor body runs.
