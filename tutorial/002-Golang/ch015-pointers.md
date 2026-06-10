# Pointers

Pointers hold memory addresses. Unlike C, Go has **no pointer arithmetic** and provides **automatic memory management** (garbage collection), making pointers safer and simpler.

```go
var a int = 42
var b *int = &a  // *int = pointer to int, &a = address of a
fmt.Println(a, b, *b)  // 42, 0x... (address), 42
```

The `&` operator gives the address of a variable. The `*` operator dereferences a pointer — it reads or writes the value at the address.

Since `b` points to the same memory location as `a`, changes through either variable affect the other:

```go
a = 10      // changes both a and *b
*b = 32     // changes both *b and a
```

## Struct Pointers

Go provides implicit dereferencing for structs — you can write `ms.foo` instead of `(*ms).foo`:

```go
type myStruct struct {
    foo int
}

var ms *myStruct
ms = new(myStruct)      // new() allocates a zero-valued struct and returns a pointer
(*ms).foo = 42          // explicit dereference — works but verbose
ms.foo = 37             // Go does it automatically — implicit dereferencing
```

The `new` keyword allocates memory for a type and returns a pointer to it. The default value for any pointer is `nil`.

## Arrays: Address Spacing

Array elements are stored contiguously in memory. Pointers to consecutive elements differ by the size of the element type (8 bytes for `int` on 64-bit systems):

```go
c := [3]int{1, 2, 3}
d := &c[0]
e := &c[1]
fmt.Printf("d: %p, e: %p\n", d, e)  // addresses differ by 8 bytes
```

## Pointers Avoid Copying Large Arrays

Arrays are value types — passing them to functions copies the entire data. Using a pointer avoids this overhead:

```go
arr := &myArray  // pass pointer instead of copying the whole array
```

However, slices are already reference types and are generally preferred over array pointers in Go.
