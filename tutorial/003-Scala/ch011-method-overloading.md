# Method Overloading

Method names can be overloaded in Scala as long as they have different **signatures**. A signature consists of the number of parameters and their types — the return type is **not** considered part of the signature.

## Overloading by Parameter Type

You can define multiple methods with the same name as long as their parameter types differ:

```scala
def add3(x: Int) = x + 3
def add3(x: Double) = x + 3.0
def add3(x: String) = x + " Three"

println(add3(3))       // 6
println(add3(5.0))     // 8.0
println(add3("One Two")) // One Two Three
```

Each version of `add3` handles a different type. The compiler picks the correct one based on the argument type at the call site.

## Return Type Is Not Part of the Signature

Two methods with the same input types but different return types **cannot** be overloaded. This will cause a compilation error:

```scala
// This will NOT compile
// def doSomething(n: Int): Int = n + 5
// def doSomething(x: Int): Double = x + 5.0
```

The compiler cannot distinguish which method to call based on return type alone, because the return type is not part of the method signature in Scala (or in Java).
