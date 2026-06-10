# Exported Names

Go determines visibility by the **first character** of a name. This is not a convention or style choice — it is part of the language spec and enforced at compile time.

- **Capital letter** = exported (public, accessible from other packages that import this package)
- **Lowercase** = unexported (private to the package, even if another package imports it)

```go
package main

import "fmt"

func main() {
    // Println starts with capital P — exported from fmt package
    fmt.Println("Hello There")
}
```

Notice `Println` starts with a capital letter. This is intentional — `fmt` is a package, and only functions/variables starting with a capital letter are accessible outside that package. If a function or package-level variable name starts with a lowercase letter, it won't be accessible even if another package imports it.

This rule applies to functions, types, variables, and constants declared at the package level.
