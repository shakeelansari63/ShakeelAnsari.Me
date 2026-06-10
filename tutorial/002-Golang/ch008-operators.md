# Operators

## Arithmetic

```go
a, b := 10, 3
fmt.Println("a + b =", a+b)  // Addition: 13
fmt.Println("a - b =", a-b)  // Subtraction: 7
fmt.Println("a * b =", a*b)  // Multiplication: 30
fmt.Println("a / b =", a/b)  // Division: 3 (integer division truncates toward zero)
fmt.Println("a % b =", a%b)  // Modulus: 1
```

Integer division truncates the decimal part — `10 / 3` gives `3`, not `3.33`.

## Bitwise

```go
fmt.Println("a & b  =", a&b)   // AND: 2 (bits set in both)
fmt.Println("a | b  =", a|b)   // OR: 11 (bits set in either)
fmt.Println("a ^ b  =", a^b)   // XOR: 9 (bits set in one but not both)
fmt.Println("a &^ b =", a&^b)  // AND NOT (bit clear): 8 (bits in a not in b)
fmt.Println("a << 3 =", a<<3)  // Left shift: 80 (a * 2^3)
fmt.Println("a >> 3 =", a>>3)  // Right shift: 1 (a / 2^3)
```

- `&` — bitwise AND
- `|` — bitwise OR
- `^` — bitwise XOR
- `&^` — bitwise NOR (AND NOT) — clears bits in `a` that are set in `b`
- `<<` — left shift (multiply by power of 2)
- `>>` — right shift (divide by power of 2)

## String Operations

```go
s1 := "I am test String1"
s2 := "I am test String 2"
fmt.Println(s1 + s2)            // Concatenation: joins both strings
fmt.Printf("%v, %T\n", s1[2], s1[2])  // Byte at index: 109, uint8
bt := []byte(s2)                // String to byte slice
```

The `+` operator concatenates strings. Indexing a string with `s1[2]` accesses the individual byte at that position — note the type is `uint8`, not `rune` or `string`, because strings are UTF-8 encoded bytes under the hood. Converting a string to `[]byte` gives you the raw byte slice.
