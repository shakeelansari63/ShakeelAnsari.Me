# Companion Objects

A **Companion Object** is an `object` with the same name as a `class` (or `trait`), defined in the same source file. The class and its companion object can access each other's private members.

## Class Accessing Companion's Private Members

Here the `SecretAgent` class accesses the companion object's private method `reduceBullet`:

```scala
case class SecretAgent(name: String) {
    def shoot(n: Int) = SecretAgent.reduceBullet(n)
}

object SecretAgent {
    private var b: Int = 3000
    private def reduceBullet(n: Int) = if (b - n < 0) b = 0 else b = b - n
    def bullets: Int = b
}

object RunnerX extends App {
    val jason = SecretAgent("Jason Bourne")
    val james = SecretAgent("James Bond")
    val jagga = SecretAgent("Jagga Jasoos")

    jason.shoot(300)
    james.shoot(500)
    jagga.shoot(200)

    println(SecretAgent.bullets) // 2000
}
```

The bullet inventory (`b`) is shared across all agents because it lives in the companion **object** (a singleton), not in the class instances. Each agent's `shoot` call reduces the same shared count.

## Companion Accessing Class's Private Members

The companion object can also access the class's private fields:

```scala
case class NormalPerson(name: String, private val superheroName: String)

object NormalPerson {
    def revealSecretIdentity(x: NormalPerson) = x.superheroName
}

object SuperHeroRunner extends App {
    val clarke = NormalPerson("Clark Kent", "Superman")
    val natasha = NormalPerson("Natasha Romanohf", "Black Widow")
    val diana = NormalPerson("Diana Prince", "Wonder Woman")

    println(NormalPerson.revealSecretIdentity(clarke))  // Superman
    println(NormalPerson.revealSecretIdentity(natasha)) // Black Widow
    println(NormalPerson.revealSecretIdentity(diana))   // Wonder Woman
}
```

Even though `superheroName` is `private`, the companion object `NormalPerson` can read it. Outside code cannot — it must go through `revealSecretIdentity`.

## Factory Pattern with Companion Objects

Companion objects are commonly used as **factories** — they create instances of the class while hiding construction details:

```scala
import java.time._

case class CompanyEmployee private (
    firstName: String,
    lastName: String,
    title: String,
    hireDate: LocalDate
)

object CompanyEmployee {
    def create(firstName: String, lastName: String, title: String, hireDate: LocalDate) =
        CompanyEmployee(firstName, lastName, title, hireDate)

    def create(firstName: String, lastName: String, title: String) =
        CompanyEmployee(firstName, lastName, title, LocalDate.now())
}

object CERunner extends App {
    val ce1 = CompanyEmployee.create("Jane", "Depp", "Programmer", LocalDate.of(2023, 10, 5))
    val ce2 = CompanyEmployee.create("Mohan", "Bhargav", "Manager")

    println(ce1) // CompanyEmployee(Jane,Depp,Programmer,2023-10-05)
    println(ce2) // CompanyEmployee(Mohan,Bhargav,Manager,2024-01-15)
}
```

The `private` modifier on the case class constructor prevents direct instantiation. Users must call `CompanyEmployee.create(...)`. The second `create` overload provides a default `hireDate` of today. This pattern gives you centralized control over object creation while keeping the factory close to the class it creates.
