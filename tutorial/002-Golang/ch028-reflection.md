# Reflection

Reflection lets you inspect and manipulate types at runtime using the `reflect` package. It is powerful but should be used sparingly — prefer type-safe code when possible.

## Inspecting Struct Fields and Tags

```go
type User struct {
    Name  string `validate:"required,min=2"`
    Email string `validate:"required,email"`
    Age   int    `validate:"min=0,max=120"`
}
```

`reflect.ValueOf(u)` gives a reflection object representing the value. From it you can enumerate fields:

```go
func validate(u interface{}) error {
    v := reflect.ValueOf(u)
    for i := 0; i < v.NumField(); i++ {
        field := v.Type().Field(i)
        value := v.Field(i)
        fieldName := field.Name
        fieldType := value.Kind()
        tags := field.Tag

        fmt.Println("Name:", fieldName, "Type:", fieldType,
            "Tags:", tags, "Value:", value)
    }
    return nil
}
```

## Custom Validation Framework

Using reflection, you can build a validation framework that reads `validate` tags and checks field values against rules:

```go
for _, rule := range strings.Split(tags.Get("validate"), ",") {
    switch {
    case strings.HasPrefix(rule, "min=") && fieldType == reflect.Int:
        minval, _ := strconv.Atoi(strings.TrimPrefix(rule, "min="))
        if int(value.Int()) < minval {
            return fmt.Errorf("%s must not be less than %d", fieldName, minval)
        }
    case rule == "required":
        if value.String() == "" {
            return fmt.Errorf("%s cannot be empty", fieldName)
        }
    case rule == "email":
        emailRegex := regexp.MustCompile(`^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-z]{2,4}$`)
        if !emailRegex.MatchString(value.String()) {
            return fmt.Errorf("%s should be valid email", fieldName)
        }
    }
}
```

Full usage:

```go
func main() {
    user := User{Name: "Shakeel", Email: "a@abc.com", Age: 34}
    if err := validate(user); err != nil {
        fmt.Println("Error:", err)
    }
}
```

This demonstrates how libraries like `go-playground/validator` work internally — using reflection to read struct tags, access field values by their type, and return validation errors based on rules encoded in the tags.
