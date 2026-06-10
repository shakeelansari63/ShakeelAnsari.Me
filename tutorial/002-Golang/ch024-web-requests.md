# Web Requests

The `net/http` package handles HTTP requests, and `net/url` provides URL parsing and construction.

## GET Request

```go
resp, err := http.Get("https://api.example.com/data")
if err != nil { panic(err) }
defer resp.Body.Close()  // always close the body — Go does not do this automatically

if resp.StatusCode != 200 {
    panic("Error in API")
}

content, err := io.ReadAll(resp.Body)  // read entire response body
fmt.Println(string(content))           // bytes → string
```

The `http.Get` call returns a response object and an error. It is critical to `defer resp.Body.Close()` immediately so the connection is returned to the pool when done. Check the status code — a non-200 response might succeed without an HTTP-level error but indicate an application failure.

## URL Parsing

`url.Parse` decomposes a URL into its parts:

```go
u, _ := url.Parse("https://example.com:8080/path?key=value")
fmt.Println("Scheme:", u.Scheme)
fmt.Println("Host:", u.Host)
fmt.Println("Port:", u.Port())
fmt.Println("RawQuery:", u.RawQuery)
```

Query parameters can be extracted as a map:

```go
qparams := u.Query()
for key, val := range qparams {
    fmt.Println(key, "=", val)
}
```

## URL Construction

Build a URL from parts using `&url.URL` (note: address-of, not value):

```go
urlParts := &url.URL{
    Scheme:   "https",
    Host:     "me.dev",
    Path:     "test",
    RawQuery: "style=1&type=2",
}
fmt.Println(urlParts.String())  // reconstruct full URL string
```
