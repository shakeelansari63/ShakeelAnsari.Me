# Higher-Order Functions

Higher-order functions are functions that take other functions as parameters or return functions as results.

## Closures

A closure is a function that captures variables from its enclosing scope. The function "closes over" the environment in which it was defined:

```scala
var factor = 2
val multiplier = (n: Int) => n * factor
println(multiplier(5))  // 10 — factor is 2

factor = 3
println(multiplier(5))  // 15 — captures the variable, not the value
```

The function `multiplier` remembers the variable `factor`, not its value at the time of creation. If `factor` changes, the function sees the new value. This is what makes it a closure — it is closed around the environment (the variable `factor`), not just the value.

```scala
class Foo(x: Int) {
    def bar(f: Int => Int) = f(x)
}

val m = 150
val f = (x: Int) => x + m  // m is captured from the enclosing object

val foo = new Foo(100)
println(foo.bar(f))  // 250 — 100 + 150
```

Even when `f` is passed to `foo.bar`, it still remembers `m = 150` from its original scope.

## Currying

Currying transforms a function that takes multiple parameters into a chain of single-parameter functions:

```scala
// Curried method defined with multiple parameter lists
def add(x: Int)(y: Int): Int = x + y
println(add(3)(4))  // 7

// Fix the first parameter, get back a function
val addFive = add(5) _
println(addFive(10))  // 15
```

Functions can also be curried using `.curried`:

```scala
val g = (x: Int, y: Int) => x + y
val curriedG = g.curried
println(curriedG(4)(5))  // 9
```

Methods can be defined with multiple parameter lists directly:

```scala
def m1(x: Int, y: Int, z: Int) = x + y + z  // normal
def m2(x: Int)(y: Int)(z: Int) = x + y + z  // fully curried
def m3(x: Int)(y: Int, z: Int) = x + y + z  // split curried

println(m2(3)(4)(5))
println(m3(3)(4, 5))
```

## Partially Applied Functions

A partially applied function is a way to convert a method into a function with some parameters already fixed:

```scala
def multiply(x: Int, y: Int): Int = x * y

// Fix x=2, leave y as a parameter
val double = multiply(2, _: Int)
println(double(5))  // 10 — 2 * 5
```

The underscore `_` indicates which parameters are left unfilled, creating a `Function1` from a `Function2`:

```scala
class Foo(x: Int) {
    def bar(y: Int) = x + y
    def baz1(y: Int, z: String) = s"$z - ${x + y}"
}

val foo = new Foo(10)

// Convert bar to function
val f1 = foo.bar _  // same as (y: Int) => foo.bar(y)
println(f1(5))  // 15

// Partially apply — fix y, leave z
val f2 = foo.baz1(4, _: String)
println(f2("Hello"))  // "Hello - 14"

// Fix z, leave y
val f3 = foo.baz1(_: Int, "Jimmy")
println(f3(1))  // "Jimmy - 11"
```

## By-Name Parameters

By-name parameters are evaluated lazily — only when they are used inside the method, not when the method is called. This allows control over when (and whether) an expression is evaluated:

```scala
def byValue(x: Int)(y: Int) = { println("Method Called"); x + y }
def byFunction(x: Int)(y: () => Int) = { println("Method Called"); x + y() }
def byName(x: Int)(y: => Int) = { println("Method Called"); x + y }

// By-Value — y is evaluated BEFORE the method body
byValue(3) { println("Y evaluated"); 20 }
// Output:
// Y evaluated
// Method Called
// 23

// By-Function — y is evaluated INSIDE the method when y() is called
byFunction(3)(() => { println("Y evaluated"); 20 })
// Output:
// Method Called
// Y evaluated
// 23

// By-Name — same lazy behavior as by-function, but cleaner syntax like by-value
byName(3) { println("Y evaluated"); 20 }
// Output:
// Method Called
// Y evaluated
// 23
```

By-name parameters use `=> Type` syntax. They are evaluated each time they are referenced in the method body, making them useful for implementing custom control structures and logging.

## Higher-Order Functions

Higher-order functions take functions as input or return functions as output:

```scala
// Function that takes a function as input
val f1 = (x: Int, f: Int => Int) => f(x)

// Passing a named function
val f2 = (x: Int) => x * 3
println(f1(3, f2))  // 9

// Passing an anonymous function
println(f1(7, (x: Int) => x * 3))  // 21

// With type inference
println(f1(5, x => x * 3))  // 15

// Using underscore shorthand
println(f1(8, _ * 3))  // 24

// Function that returns a function
val g = (x: Int) => (y: Int) => x + y
val g1 = g(3)  // g1 is a function: Int => Int
println(g1(4))  // 7
println(g(3)(9))  // 12 — calling the curried function
```
