# If, Switch & Type Switch

## If-Else

Go's `if` follows the standard `if...else if...else` ladder with no parentheses required around the condition:

```go
num := 6
if num <= 5 {
    fmt.Println("Less than five")
} else if num <= 10 {
    fmt.Println("Less than 10")
} else {
    fmt.Println("Greater than 10")
}
```

Multiple conditions can be combined with `&&` (AND) and `||` (OR):

```go
if num <= 5 && num >= 2 {
    fmt.Println("Between 2 and 5")
} else if num > 5 || (num < 2 && num >= 0) {
    fmt.Println("Outside range")
}
```

Go uses **short-circuit evaluation**: for `||`, evaluation stops as soon as a `true` operand is found; for `&&`, it stops as soon as a `false` operand is found.

## If with Initializer

A powerful Go feature is the `if` statement with an initializer. Variables declared in the initializer are **scoped to the if-else block** — they do not leak to the surrounding scope. This also lets you redeclare the same variable names in different branches:

```go
mp := map[string]int{"key1": 10, "key2": 20}
if pop, ok := mp["key4"]; ok {
    fmt.Println("key4:", pop)
} else if pop, ok := mp["key2"]; ok {
    fmt.Println("key2:", pop)
}
```

The variables `pop` and `ok` are only visible inside each branch. This is why you can reuse them across branches without conflict.

## Switch

Go's `switch` works similarly to C, but with one critical difference: **there is no implicit fallthrough**. Each case automatically breaks — you do not need a `break` statement:

```go
switch num {
case 1:
    fmt.Println("One")
case 2:
    fmt.Println("Two")
default:
    fmt.Println("None")
}
```

### Multi-Case Switch

Multiple values can be matched in a single case:

```go
switch num {
case 1, 3, 5:
    fmt.Println("odd")
case 2, 4, 6:
    fmt.Println("even")
}
```

### Switch with Initializer

Like `if`, `switch` also supports an initializer that is scoped to the switch block:

```go
switch num2 := (num / 2) + 3; num2 {
case 1, 3, 5:
    fmt.Println("odd")
case 2, 4, 6:
    fmt.Println("even")
}
```

### Inequality Switch

By omitting the expression after `switch`, each case can test an arbitrary boolean condition:

```go
switch {
case num%2 == 0:
    fmt.Println("Even")
case num%2 != 0:
    fmt.Println("Odd")
}
```

### Fallthrough

Though Go does not fall through by default, you can explicitly request it with the `fallthrough` keyword. This is rarely used but exists for rare cases where you want overlapping case behaviour:

```go
switch {
case num <= 10:
    fmt.Println("<= 10")
    fallthrough
case num <= 20:
    fmt.Println("<= 20")
}
```

### Type Switch

A type switch lets you inspect the dynamic type of an interface value:

```go
var num3 any = 2
switch num3.(type) {
case int, int8, int16, int32, int64:
    fmt.Println("integer")
case float32, float64:
    fmt.Println("float")
case string:
    fmt.Println("string")
default:
    fmt.Println("other")
}
```

The `.`**(type)** syntax is only valid inside a `switch` statement. It checks what concrete type is stored in the interface variable.
