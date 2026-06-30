# Maps & Structs

## Maps

Maps are key-value collections with a fixed key type and fixed value type. They are **reference types** — assigning a map to another variable shares the underlying data. Unlike arrays or slices, maps are **not ordered**.

```go
mp1 := map[string]int{"key1": 10, "key2": 20, "key3": 30}
```

### Creating Maps

Maps can also be created with `make`, which is useful when you want to add entries dynamically:

```go
mp2 := make(map[string]int)
mp2["k1"] = 2
mp2["k2"] = 7
```

### Access and Delete

Access individual values with bracket notation. Use `delete` to remove a key:

```go
fmt.Println(mp2["k2"])
delete(mp1, "key2")
```

### The Comma-Ok Idiom

Accessing a key that doesn't exist returns the zero value for the value type — it does not error. To distinguish between "key exists with zero value" and "key doesn't exist", use the comma-ok idiom:

```go
value, ok := mp1["key2"]
if !ok {
    fmt.Println("key doesn't exist")  // key2 was deleted
}
```

### Length

```go
fmt.Println(len(mp1))
```

### Reference Semantics

Maps are reference types. Assigning one map variable to another does not copy the data — both point to the same underlying map:

```go
mp3 := mp1
mp3["key7"] = 32  // mp1 also sees this change
```

## Structs

Structs collect related fields into a single type. They are **value types** — assignment creates a copy.

```go
type MyStruct struct {
    id      int    `required max:999`
    name    string
    friends []string
}

st1 := MyStruct{id: 1, name: "Dino", friends: []string{"Milo", "Jeff"}}
fmt.Println(st1.name)
fmt.Printf("%+v\n", st1)  // %+v prints field names and values
```

### Anonymous Structs

For one-off uses, you can define a struct inline without a named type:

```go
st2 := struct {
    name string
    age  int
}{name: "Shaw", age: 32}
```

### Composition (Embedding)

Go does not have classical inheritance. Instead, it supports composition through struct embedding. When you embed one struct inside another, the outer struct gains direct access to all the inner struct's fields:

```go
type MyStruct2 struct {
    MyStruct          // embedded — fields promoted to outer struct
    age int
}
st3 := MyStruct2{}
st3.id = 2
st3.name = "Julie"
st3.friends = []string{"Dino", "Jeff"}
st3.age = 21
```

### Struct Tags with Reflection

Struct fields can have **tags** — metadata strings that are invisible to normal code but accessible via reflection. These are commonly used for validation, serialization, and ORM mapping:

```go
t := reflect.TypeOf(MyStruct{})
field, _ := t.FieldByName("id")
fmt.Println(field.Tag)  // "required max:999"
```

Tags are just strings — they have no built-in behaviour. Libraries like `go-playground/validator` and `encoding/json` read them via the `reflect` package to drive their behaviour.

### Summary

- **Structs** are value types — assignment creates a full copy.
- **Maps** (and slices) are reference types — assignment shares the underlying data.
