# Methods

Methods are functions with a **receiver** argument. Unlike class-based languages where methods are defined **inside** the class, Go defines methods **outside** the struct. The receiver connects the function to the type.

```go
type User struct {
    Name  string
    Age   int
    Email string
}

// Value receiver — operates on a copy of the struct
func (u User) Greet() {
    fmt.Printf("Hello I am %s, %d years old\n", u.Name, u.Age)
}

// Value receiver cannot modify the original
func (u User) UpdateEmail() {
    u.Email = "test@go.dev"
    fmt.Println("New email:", u.Email)
}

func main() {
    usr := User{"Shakeel", 31, "shakeel@go.dev"}
    usr.Greet()
    usr.UpdateEmail()  // modifies the copy, not the original
    usr.Greet()        // Email is still "shakeel@go.dev"
}
```

## Value Receiver vs Pointer Receiver

A **value receiver** receives a copy of the struct. Changes inside the method do not affect the caller's original. This is safe and predictable.

A **pointer receiver** receives a pointer to the struct. Changes affect the original:

```go
func (u *User) SetAge(age int) {
    u.Age = age  // modifies the original
}
```

Use a pointer receiver when:
- You need to mutate the receiver
- The struct is large — copying it would be expensive

Go automatically converts between value and pointer receivers. If you have a pointer and call a value-receiver method, Go dereferences it for you. If you have a value and call a pointer-receiver method, Go takes its address automatically (as long as the value is addressable).

## Key Insight

Unlike languages like Java or C++, Go methods are not defined inside the type declaration. The receiver is just an additional parameter placed before the function name. This means you can add methods to **any type** defined in your package, not just structs.
