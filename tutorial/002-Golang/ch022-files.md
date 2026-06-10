# File I/O

File operations in Go use multiple packages: `os` for creating/opening files, `io` for writing, and `os` for reading.

```go
package main

import (
    "fmt"
    "io"
    "os"
)

func main() {
    fileName := "./my-testing-file.txt"

    // Create file — returns a file pointer and an error
    file, err := os.Create(fileName)
    handleError(err)

    // Defer close immediately after opening — best practice to never forget
    defer file.Close()

    // Write string content — returns bytes written and error
    length, err := io.WriteString(file, "Hello world!!!")
    handleError(err)
    fmt.Println("Written:", length)

    // Read entire file — returns []byte and error
    readContent(fileName)
}

func readContent(fl string) {
    content, err := os.ReadFile(fl)
    handleError(err)
    fmt.Println(string(content))  // bytes → string for human-readable output
}

func handleError(err error) {
    if err != nil {
        panic(err)
    }
}
```

## Key Patterns

**Defer close immediately**: As soon as a file is successfully opened, defer the close on the next line. This guarantees cleanup regardless of how the function exits — whether normally, via an error return, or even a panic.

**`os.ReadFile` returns bytes**: The file content is read as `[]byte`. To display it as text, convert to string with `string(content)`.

**Error handling**: File operations commonly fail (permissions, missing files, disk full). The standard Go pattern is to check the error immediately and handle it, often by panicking in simple programs or by returning the error in libraries.

## Alternative: Write with `io.WriteString`

Prints the length of content actually written to the file, which is useful for confirming the write succeeded or detecting partial writes.
