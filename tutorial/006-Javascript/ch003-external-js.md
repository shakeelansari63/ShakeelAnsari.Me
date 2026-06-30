# External JavaScript Files

For larger projects, keep JavaScript in separate `.js` files and link them in HTML.

## Creating an External Script

**script.js:**
```javascript
console.log("Hello from external file!");

function greet(name) {
    return "Hello, " + name + "!";
}
```

## Linking in HTML

```html
<!DOCTYPE html>
<html>
<head>
    <title>External JS</title>
    <script src="script.js" defer></script>
</head>
<body>
    <h1>Open the console</h1>
</body>
</html>
```

## Benefits of External Files

- **Separation of concerns** — HTML structure vs JavaScript behavior
- **Caching** — browsers cache external JS files across pages
- **Maintainability** — easier to manage and debug
- **Reusability** — same script can be used on multiple pages

## Multiple Scripts

```html
<script src="lib.js" defer></script>
<script src="app.js" defer></script>
```

With `defer`, scripts execute in the order they appear in the HTML.

## Script Loading Strategies

| Attribute | Download | Execution | Order |
|-----------|----------|-----------|-------|
| (none, in `<head>`) | Blocks parsing | Immediately | In order |
| (none, end of `<body>`) | After HTML parsed | After HTML parsed | In order |
| `defer` | In parallel | After HTML parsed | In order |
| `async` | In parallel | As soon as downloaded | First-come |
