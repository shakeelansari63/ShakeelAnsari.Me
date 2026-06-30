# Methods

Methods in Scala are defined with the `def` keyword. The last evaluated statement's value is always returned from the method — there is no need for an explicit `return` keyword.

```scala
def add(a: Int, b: Int): Int = a + b
println(add(6, 7))
```

## Single-Expression Methods

Since `if` also returns a value, you can use it directly in a method body:

```scala
def compare(a: Int, b: Int) = if (a > b) "Greater"
                              else if (a < b) "Smaller"
                              else "Same"
```

## Methods with Multiple Return Types

Methods can return different types depending on the branch:

```scala
def intOrString(a: Int, b: Int) = if (a > b) a + b
                                   else (a + b).toString
```

The return type is inferred as the common supertype — in this case, `Any`.

## Nested Methods

Methods can be defined inside other methods. This is useful when you need a helper that should only be accessible within the outer method:

```scala
def someMethod(n: Int): Int = {
    def otherMethod(n: Int): Int = {
        n * 2
    }
    otherMethod(n)
}
println(someMethod(3))
```

`otherMethod` is scoped to `someMethod` and cannot be accessed from outside.

## Special Characters in Names

Like variables, method names can contain special characters if enclosed in backticks or if special symbols are preceded by an underscore:

```scala
// Backticks allow spaces and reserved words
def `Special method 1`(n: Int) = n + 2

// Underscore before special characters
def areYouHappy_?() = true
println(if (areYouHappy_?) "Nice" else "Ugh!!")

// Reserved words can be used as names in backticks
def `return`(n: Int) = n + 5
```
