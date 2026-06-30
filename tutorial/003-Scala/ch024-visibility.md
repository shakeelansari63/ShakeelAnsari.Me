# Visibility Options

Scala has three main visibility levels:

| Option | Default? | In class? | Companion Object? | Subclass? | Package? | Everywhere? |
|--------|----------|-----------|-------------------|-----------|----------|-------------|
| Public | Yes | Yes | Yes | Yes | Yes | Yes |
| Private | No | Yes | Yes | No | No* | No |
| Protected | No | Yes | Yes | Yes | No* | No |

## Scoped Visibility

The `*` on Private and Protected means these can be scoped to a specific package. You can specify the scope in brackets after `private` or `protected`:

```scala
package org.scala.test

case class X(private[scala] val x: Int)
// x is visible inside the entire org.scala package
```

So if there are other classes in `org.scala`, they can access `x`:

```scala
package org.scala.test2

val x = X(2)
println(x.x)  // Possible — same org.scala package
```

But classes outside the `org.scala` package cannot:

```scala
package org.scala3.test3

val x = X(2)
// println(x.x)  // Not possible — different package
```

## Instance-Level Private

`private[this]` makes a variable private not just to the class, but to the specific instance. This means one instance cannot access the private field of another instance of the same class:

```scala
case class Y(private[this] val x: Int) {
    // def update(other: Y) = x = other.x  // Error
    // Cannot access other.x — private[this] is instance-level
}
```

This provides the strongest level of encapsulation in Scala.
