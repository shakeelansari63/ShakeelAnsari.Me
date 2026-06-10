# Variables

Go is statically typed — every variable has a fixed type known at compile time. Variables can be declared at package scope (with `var`) or inside functions (with `var` or `:=`).

## Declaration Forms

```go
var num1 int              // zero value (0) — default for int is 0
var num2 int = 3          // declare and initialize
var num3 = 2              // type inference — Go identifies the type from the value
num6 := 8                 // shorthand — only inside functions, same as var num6 = 8
```

If you declare a variable but don't use it, Go gives a compile error. Unused variables are not allowed.

Package-level variables must use `var` — the `:=` syntax is only valid inside functions.

```go
var globalVar int = 7
```

## Numeric Types

Go has many integer types, each with a specific range:

- `int8`: -128 to 127
- `int16`: -32768 to 32767
- `int32`: -2147483648 to 2147483647
- `int64`: -9.22e18 to 9.22e18
- `int`: depends on system architecture (32-bit on 32-bit systems, 64-bit on 64-bit)

There are corresponding unsigned variants: `uint8`, `uint16`, `uint32`, `uint64`, `uint`. `byte` is an alias for `uint8`.

Float types: `float32` and `float64`. Default is `0.0`.

```go
var num8 float32 = 10.5
var num9 float64 = 19.7
```

Complex numbers come as `complex64` (float32 real+imag parts) and `complex128` (float64 parts). Default is `0+0i`.

```go
var com1 complex64 = 10 + 5i
var com2 complex128 = 19 + 7.5i
fmt.Printf("Real: %f Imag: %f\n", real(com1), imag(com1))
```

The `real()` and `imag()` built-in functions extract the real and imaginary parts.

## Strings, Runes & Booleans

Strings are UTF-8 characters enclosed in double quotes `""` or backticks ```` ``` ``. Default is `""`.

```go
var str1 string = "Hello World"
str2 := `raw string`
```

A rune is a single UTF-32 character defined in single quotes:

```go
var rn rune = 'a'
fmt.Printf("Rune : %v, %T\n", rn, rn)
```

Booleans use the `bool` keyword. Default is `false`.

```go
bl1 := true
bl2 := false
```

Use `%T` in formatted print to inspect a variable's type:

```go
fmt.Printf("Data Types: %T %T %T %T\n", str1, num1, num8, bl1)
```

## Multi-Variable Declaration

You can declare multiple variables of the same type in one line:

```go
var num4, num5 int
```

But you cannot mix types in a single `var` line. Instead, use a var block:

```go
var (
    myvar1 int    = 2
    myvar2 string = "myvar"
    myvar3 bool   = true
)
```

Multiple values can be assigned to multiple variables in one line:

```go
num4, num5 = 6, 9
```

## Type Conversion

Go has no implicit type casting. Even types as similar as `int` and `int8` must be explicitly converted:

```go
intDat := 42
var flDat float32 = float32(intDat)
```

Directly converting an `int` to `string` does not produce digits — it treats the int as a Unicode code point:

```go
strDat := string(intDat)  // "*" — Unicode 42 is asterisk
```

To get the actual decimal string representation, use `strconv.Itoa`:

```go
strDat = strconv.Itoa(intDat)  // "42"
```
