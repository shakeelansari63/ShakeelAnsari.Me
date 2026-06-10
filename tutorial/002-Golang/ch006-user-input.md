# User Input

Reading user input from the terminal requires a buffered reader wrapping standard input.

```go
package main

import (
    "bufio"
    "fmt"
    "os"
)

func main() {
    reader := bufio.NewReader(os.Stdin)

    fmt.Println("Enter a number: ")
    num, _ := reader.ReadString('\n')
    fmt.Println("Your Number is:", num)

    x, err := reader.ReadString('\n')
    fmt.Printf("Value: %s, Error: %v\n", x, err)
}
```

## How It Works

`bufio.NewReader(os.Stdin)` wraps the operating system's standard input in a buffered reader for efficient reading. The `ReadString` method reads input until it encounters the delimiter (`\n` — newline).

A key Go philosophy is visible here: `ReadString` returns **two values** — the string entered by the user and an error object. Instead of raising exceptions like many languages, Go returns errors as values that you can check and handle explicitly.

## The Comma-Ok Pattern

If you want to ignore one of the return values, use the blank identifier `_`:

```go
num, _ := reader.ReadString('\n')
```

This is called the "comma ok" syntax. The second returned value is discarded.

To see what the error object looks like, capture it:

```go
x, err := reader.ReadString('\n')
fmt.Printf("Value: %s, Error: %v\n", x, err)
```

When there is no error, `err` will be `nil`. When something goes wrong (like an EOF), `err` will contain details.
