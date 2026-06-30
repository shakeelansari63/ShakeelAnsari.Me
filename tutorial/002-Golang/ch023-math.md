# Math

The `math` package provides mathematical constants and functions.

```go
num := 12.497
sqr := math.Sqrt(num)
fmt.Printf("Square root: %.2f\n", sqr)
```

## Formatting: `%g` vs `%f`

- `%g` — rounds the output, removing trailing zeros
- `%f` — prints the exact value with the specified precision

```go
fmt.Printf("Square root of %g is %.2f \n", num, sqr)
```

## Rounding

```go
fmt.Println("Round:", math.Round(sqr))   // nearest integer
fmt.Println("Floor:", math.Floor(sqr))   // down to nearest integer
fmt.Println("Ceil:", math.Ceil(sqr))     // up to nearest integer
```

## Constants

```go
fmt.Println("E:", math.E)
```

## Random Numbers

Since Go 1.20, the global random source is automatically seeded. For older Go versions, seed explicitly:

```go
// Go <1.20 only — auto-seeded in Go 1.20+
// rand.Seed(time.Now().UnixNano())
fmt.Println("Random [0-5]:", rand.Intn(6))  // Intn(n) returns [0, n)
```

Without explicit seeding in older versions, `rand` produces the same sequence every run. Since Go 1.20, this is no longer necessary.
