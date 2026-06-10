# Entry Points (Main Method)

Scala programs need an entry point to start execution, similar to Java's `public static void main(String[] args)`. There are two ways to define one.

## Using a `main` Method

Define an `object` (not a class) with a `main` method that takes `Array[String]` as its parameter:

```scala
object Runner {
    def main(args: Array[String]) = println("Hello from Scala")
}
```

Because `Runner` is an `object` (a singleton), its members are static. When compiled with `scalac` and inspected with `javap`, you'll see:

```
public final class Runner {
  public static void main(java.lang.String[]);
}
```

The Scala compiler generates a static `main` method that Java's runtime can find and invoke.

## Using `extends App`

If you don't want to write the `main` method boilerplate, extend `App`:

```scala
object Runner2 extends App {
    println("Hello from scala.App object")
}
```

The `App` trait provides a delayed initialization mechanism — any code inside the object body becomes the program's entry point. Compiling and inspecting with `javap` reveals:

```
public final class Runner2 {
  public static void main(java.lang.String[]);
  public static void delayedInit(scala.Function0<scala.runtime.BoxedUnit>);
  public static long executionStart();
}
```

`App` generates the `main` method automatically, plus `delayedInit` (which executes the body) and `executionStart` (a timestamp for measuring program duration).

The `extends App` approach is more concise and is preferred for simple scripts and applications, while the explicit `main` method gives you more control and matches the familiar Java pattern.
