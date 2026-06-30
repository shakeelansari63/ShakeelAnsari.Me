# Inline JavaScript in HTML

JavaScript can be written directly inside an HTML file using the `<script>` tag.

## Basic Inline Script

```html
<!DOCTYPE html>
<html>
<head>
    <title>Inline JS</title>
</head>
<body>
    <h1>Check the console</h1>

    <script>
        console.log("Hello from inline JavaScript!");
        alert("This is an alert box");
    </script>
</body>
</html>
```

## Placement Matters

Scripts in the `<head>` block rendering until they execute. Scripts at the end of `<body>` let the page render first.

```html
<head>
    <script>
        // Runs before the page is fully rendered
        console.log("Head script");
    </script>
</head>
<body>
    <h1>Page content</h1>

    <script>
        // Runs after the body content is parsed
        console.log("Body script");
    </script>
</body>
```

## Using `defer` and `async`

```html
<script defer src="script.js"></script>
<!-- defer: download in parallel, execute after HTML parsed, in order -->

<script async src="script.js"></script>
<!-- async: download in parallel, execute as soon as downloaded -->
```
