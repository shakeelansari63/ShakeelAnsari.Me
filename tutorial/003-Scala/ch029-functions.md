# Functions

Functions in Scala are objects of classes `Function0` through `Function22` — meaning a function can take up to 22 parameters. Each `FunctionX` class has an `apply` method that is called when the function is invoked.

## Creating Functions Explicitly

```scala
val f1: Function0[Int] = new Function0[Int] {
    def apply(): Int = 1
}

val f2: Function1[Int, Int] = new Function1[Int, Int] {
    def apply(x: Int): Int = x + 5
}

val f3: Function3[Int, Int, String, String] =
    new Function3[Int, Int, String, String] {
        def apply(x: Int, y: Int, z: String): String = s"$z - $x $y"
    }

// Call via apply
println(f1.apply())          // 1
println(f2.apply(3))         // 8
println(f3.apply(2, 3, "Hi")) // "Hi - 2 3"

// Syntactic sugar — Scala lets you omit .apply()
println(f1())     // 1
println(f2(3))    // 8
println(f3(2, 3, "Hi"))  // "Hi - 2 3"
```

## Simplified Syntax

You don't need to instantiate `FunctionX` classes directly. There is a cleaner syntax using `=>`:

```scala
val f4: () => Int                     = () => 1
val f5: (Int) => Int                  = (x: Int) => x + 5
val f6: (Int, Int, String) => String  = (x: Int, y: Int, z: String) => s"$z - $x $y"

println(f4())
println(f5(3))
println(f6(2, 3, "Hi"))
```

With type inference, you can simplify even further:

```scala
val f7 = () => 1
val f8 = (x: Int) => x + 5
val f9 = (x: Int, y: Int, z: String) => s"$z - $x $y"

println(f7())
println(f8(3))
println(f9(2, 3, "Hi"))
```

## Methods vs Functions

| Aspect | Method | Function |
|--------|--------|----------|
| Definition | `def` inside a class/object/trait | Literal with `=>` or `FunctionX` instance |
| Nature | Not an instance — belongs to a context | Object of `FunctionX` — can be assigned, passed, returned |
| Call syntax | `obj.method(args)` | `function(args)` or `function.apply(args)` |
| `apply` | No `.apply` method | Has `.apply` method |

Methods belong to a class, object, or trait — they are not first-class values. Functions are objects and can be stored in variables, passed as arguments, and returned from other functions.
