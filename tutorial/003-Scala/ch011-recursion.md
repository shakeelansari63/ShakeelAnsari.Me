# Recursion

Recursion is a technique where a method calls itself. Since a recursive method references itself, Scala cannot always infer its return type — so a **mandatory explicit return type** is required.

## Basic Recursion

```scala
def factorial(n: Int): Int = {
    if (n == 0 || n == 1) 1
    else n * factorial(n - 1)
}

println(factorial(5))    // 120
println(factorial(100))  // 0 — numeric overflow
```

`factorial(100)` returns `0` because `Int` cannot hold such a large number — it overflows.

## BigInt for Large Numbers

To avoid numeric overflow, use `BigInt`:

```scala
def bigFactorial(n: BigInt): BigInt = {
    if (n == 0 || n == 1) 1
    else n * bigFactorial(n - 1)
}

println(bigFactorial(100))   // works
println(bigFactorial(1000))  // works
// println(bigFactorial(10000))  // stack overflow!
```

Even with `BigInt`, deep recursion causes a stack overflow because Scala uses the call stack for recursion, and the stack has a limited size.

## Tail Recursion

To overcome stack overflow, use **tail recursion** — where the recursive call is the very last operation in the method. With tail recursion, Scala optimizes the recursion into a loop, preventing stack growth.

The `@tailrec` annotation tells the compiler to verify that the method is truly tail-recursive:

```scala
import scala.annotation.tailrec

@tailrec
def tailrecFactorial(n: BigInt, accum: BigInt): BigInt = {
    if (n == 0 || n == 1) accum
    else tailrecFactorial(n - 1, accum * n)
}

println(tailrecFactorial(5, 1))
println(tailrecFactorial(10000, 1))  // no stack overflow!
```

Notice the accumulator parameter `accum`. In the non-tail-recursive version, we multiplied `n * factorial(n - 1)` — the multiplication happened *after* the recursive call returned. To make it tail-recursive, we pass the accumulated product forward so that the recursive call is the last thing executed. When `n` reaches 0 or 1, the accumulated value is returned directly.

Without `@tailrec`, if the method is not actually tail-recursive, the compiler will not optimize it and you risk stack overflow. The annotation makes the compiler enforce the optimization.
