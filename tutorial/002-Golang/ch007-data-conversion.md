# Data Conversion

When data comes in as a string (for example, from user input), you cannot perform numeric operations on it directly. The `strconv` package converts string representations to numeric types.

```go
package main

import (
    "bufio"
    "fmt"
    "os"
    "strconv"
    "strings"
)

func main() {
    n1s := "32"
    n1, _ := strconv.ParseFloat(n1s, 32)
    fmt.Println("Added 1:", n1+1)
```

## Error Handling

`strconv.ParseFloat` returns `(float64, error)`. If the string cannot be parsed as a number, the error value tells you what went wrong:

```go
    n2, err := strconv.ParseFloat("ab", 32)
    if err != nil {
        fmt.Println("Error:", err)
    }
```

This is Go's standard error handling pattern — always check `err != nil` before using the result.

## The Stdin Newline Trap

When reading from standard input, `ReadString` includes the trailing newline character `\n` in its output. Attempting to parse `"42\n"` as a float will fail:

```go
    reader := bufio.NewReader(os.Stdin)
    fmt.Print("Enter a number: ")
    input, _ := reader.ReadString('\n')
    val, err := strconv.ParseFloat(strings.TrimSpace(input), 64)
    if err != nil {
        fmt.Println("Error:", err)
    } else {
        fmt.Println("Result:", val+1)
    }
}
```

The fix is `strings.TrimSpace(input)` which strips the newline (and any other whitespace) before parsing.

## Key Functions

- `strconv.ParseFloat(s, bitSize)` — returns `(float64, error)`. `bitSize` can be 32 or 64.
- `strconv.ParseInt(s, base, bitSize)` — returns `(int64, error)`.
- `strconv.Atoi(s)` — shortcut for `ParseInt(s, 10, 0)` (string to int).
- Always trim stdin input: `strings.TrimSpace(input)`.
