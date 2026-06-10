# JSON

The `encoding/json` package provides full JSON encoding and decoding support through Go's standard library.

## Struct Tags

Struct tags configure how fields are serialized to JSON. The `json:"name"` syntax controls the JSON field name:

```go
type Course struct {
    Name     string   `json:"coursename"`          // JSON field: "coursename"
    Price    int      `json:"price"`               // JSON field: "price"
    Site     string   `json:"website"`             // JSON field: "website"
    Password string   `json:"-"`                   // excluded from JSON entirely
    Tags     []string `json:"tags,omitempty"`       // omitted if nil/empty
}
```

- `json:"-"` — field is never included in JSON
- `json:"tags,omitempty"` — field is omitted if it has its zero value (nil for slices)

## Encoding (Struct → JSON)

```go
coursesList := []Course{
    {"React JS", 299, "LearnCodeOnline.in", "abc123", []string{"web", "js"}},
    {"Angular JS", 199, "LearnCodeOnline.in", "abcd12", nil},
}

// Compact JSON — good for transmission
content, _ := json.Marshal(coursesList)
fmt.Println(string(content))

// Pretty-printed JSON — good for human reading
content, _ = json.MarshalIndent(coursesList, "", "\t")
fmt.Println(string(content))
```

`json.MarshalIndent` uses a prefix (empty string) and an indent string (`"\t"`) to format the output. The password field is excluded, and `nil` tags are omitted.

## Decoding (JSON → Struct)

```go
jsonData := []byte(`{
    "coursename": "MERN",
    "price": 299,
    "tags": ["full-stack", "js"]
}`)

// Validate JSON before decoding
fmt.Println("Is valid JSON?", json.Valid(jsonData))

// To struct — field names must match json tags
var course Course
json.Unmarshal(jsonData, &course)
fmt.Printf("%+v\n", course)

// To map — when you don't have a predefined struct
var data map[string]interface{}
json.Unmarshal(jsonData, &data)
for k, v := range data {
    fmt.Printf("Key: %v, Value: %v\n", k, v)
}
```

## Struct vs Map for Decoding

- **Struct**: type-safe, field-checked at compile time, but requires a predefined type
- **Map[string]interface{}**: flexible for dynamic data, but requires type assertions to access values

Use `json.Valid()` before unmarshalling to avoid panics on malformed JSON.
