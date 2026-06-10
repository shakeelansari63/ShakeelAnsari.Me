# Gin Web Framework

[Gin](https://github.com/gin-gonic/gin) is a high-performance HTTP web framework for Go. It provides routing, middleware, JSON validation, and error handling out of the box.

## Setup

```go
go get -u github.com/gin-gonic/gin
```

Initialize a module and tidy:

```bash
go mod init example.com/myapp
go mod tidy
```

Gin uses Go modules (introduced in Go 1.11). After `go mod tidy`, all dependencies are recorded in `go.mod` and `go.sum`.

## Minimal Server

```go
package main

import (
    "github.com/gin-gonic/gin"
)

func main() {
    r := gin.Default()
    r.GET("/", func(c *gin.Context) {
        c.JSON(200, gin.H{
            "message": "pong",
        })
    })
    r.Run() // listen on 0.0.0.0:8080
}
```

`gin.Default()` creates a router with built-in middleware (logging and recovery). Routes are registered with method-specific helpers like `r.GET`, `r.POST`, etc.

The handler function receives a `*gin.Context` which provides request parsing, response writing, and parameter access. `c.JSON` serializes a value to JSON and sets the `Content-Type` header automatically.

`gin.H` is a shortcut for `map[string]interface{}` — a convenient way to build JSON responses inline.

`r.Run()` starts the server on `0.0.0.0:8080`. You can also specify a custom address with `r.Run(":4000")`.

## Route Parameters

```go
r.GET("/user/:name", func(c *gin.Context) {
    name := c.Param("name")
    c.JSON(200, gin.H{"user": name})
})
```

`c.Param` extracts named segments from the URL pattern.

## Query Parameters

```go
r.GET("/search", func(c *gin.Context) {
    query := c.Query("q")
    c.JSON(200, gin.H{"query": query})
})
```

Gin also provides `c.DefaultQuery("key", "default")` for optional params with a fallback.

## Groups & Middleware

Routes can be grouped to share middleware:

```go
v1 := r.Group("/api/v1")
v1.Use(gin.BasicAuth(gin.Accounts{"admin": "secret"}))
{
    v1.GET("/users", listUsers)
    v1.POST("/users", createUser)
}
```

Gin is the most widely used Go web framework, chosen for its speed, minimal API surface, and extensive middleware ecosystem.
