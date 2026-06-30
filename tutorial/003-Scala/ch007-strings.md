# Strings

In Scala, strings are values enclosed in double quotes.

```scala
val a: String = "Hello"
```

## Multi-line Strings

For multi-line strings, you can concatenate lines with `\n`:

```scala
val b: String = "Hello \n" +
    "Hi \n" +
    "Buh Bye"
```

But Scala also provides **smart strings** using triple quotes `""" """`, which allow multi-line strings more naturally:

```scala
val c: String = """Hello
                   Hi
                   Buh Bye"""
```

However, the leading spaces on each line become part of the string. Use `stripMargin` with a margin character (commonly `|`) to remove them:

```scala
val d: String = """Hello
                   |Hi
                   |Buh Bye""".stripMargin('|')
```

## Regex with Smart Strings

Smart strings do not interpret backslash `\` as an escape character, which makes regular expressions much cleaner. Compare the two approaches:

```scala
val data = "Hello Its 03:00 PM"

// Regular string — need double backslashes everywhere
val regex1 = "\\s*\\d?\\d:\\d\\d\\s*(AM|PM)".r
println(regex1.findAllIn(data).toList)

// Smart string — no backslash escaping needed
val regex2 = """\s*\d?\d:\d\d\s*(AM|PM)""".r
println(regex2.findAllIn(data).toList)
```

Both produce the same result, but the smart string version is far more readable.

## String Formatting

Use `.format` on a string for positional parameter replacement:

```scala
println("Hey there, this is %s".format("Test"))
```

Positions can be specified explicitly with `%{position}${format}`:

```scala
println("Hey there... %3$s, %2$s, %1$s ... Go".format("One", "Two", "Three"))
// Prints: "Hey there... Three, Two, One ... Go"
```

Date/time formatting works with `%t{format}`:

```scala
import java.time._
println("Today is %1$te %1$tB of year %1$tY".format(LocalDate.now))
```

## String Interpolation

Scala has two string interpolators:

**s-string** — basic interpolation without extensive formatting options:

```scala
val age: Int = 32
println(s"Your age is $age")
```

**f-string** — combines s-string interpolation with format specifiers, like C-style `printf`:

```scala
val age: Int = 32
println(f"Your age is $age%1.2f")
```

## String Functions

```scala
val f = "Hello this is a simple string !!!"

// Character at position
println(f.charAt(3))              // 'l'

// Concatenate
println("Hello".concat(" Scala"))

// Check if contains substring
println("Hello".contains("ell"))

// Index of character/string
println("Hello".indexOf("ell"))

// Replace
println("Hello Scala".replace("Scala", "Java"))

// Substring (start, end) — substring is deprecated since 2.13, use slice instead
println("Hello Scala Devs".slice(6, 11))

// Trim leading and trailing spaces
println("   trim me   ".trim)
```
