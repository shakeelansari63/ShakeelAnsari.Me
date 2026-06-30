# Functions

Functions are first-class citizens in Go. Return types are declared after parameters, and Go supports multiple return values, named returns, variadic parameters, and functions as types.

## Multiple Return Values

```go
func add(x int, y int) int {
    return x + y
}

// If adjacent parameters share a type, you can omit the type for all but the last
func multipleOps(x, y int) (int, int, int) {
    return x + y, x - y, x * y
}

// Named return values — variables are initialized to their zero values
func ohMyFunction(x, y int) (sum, diff int) {
    sum = x + y
    diff = x - y
    return  // bare return — returns the named values
}
```

Named return values create local variables initialized to zero. A bare `return` returns their current values. This can make code clearer but should be used judiciously.

## Call by Value vs Call by Reference

Everything in Go is passed **by value** — the function receives a copy. Slices and maps contain references to underlying data, so changes to elements inside the function affect the caller, but reassigning the slice header itself does not propagate.

```go
func callByValue(x, y string) {
    x = "Ted"  // only changes the local copy; original unchanged
}
func callByReference(x, y *string) {
    *x = "Ted"  // dereferences the pointer, changes the original
}
```

## Variadic Functions

The `...` syntax packs an arbitrary number of arguments into a slice:

```go
func numSum(values ...int) {
    sum := 0
    for _, v := range values {
        sum += v
    }
    fmt.Println(sum)
}
numSum(1, 2, 3)
numSum(1, 2, 3, 4, 5)
```

## Returning Errors

A common Go pattern is returning a value and an error. The caller checks the error before using the value:

```go
func numDiv(x, y float32) (float32, error) {
    if y == 0.0 {
        return 0.0, fmt.Errorf("cannot divide by zero")
    }
    return x / y, nil
}

val, err := numDiv(10., 0.)
if err != nil {
    fmt.Println("Error:", err)
} else {
    fmt.Println("Result:", val)
}
```

## Anonymous Functions and Function Types

Functions can be assigned to variables, passed as arguments, and returned from other functions:

```go
anomFunc := func() {
    fmt.Println("Hello World")
}
anomFunc()  // call it like a regular function

// Passing a function as a parameter
getFunc(anomFunc)

func getFunc(x func()) {
    x()
}
```

This makes higher-order functions possible — you can write functions that accept or return other functions, enabling patterns like callbacks, decorators, and middleware.
