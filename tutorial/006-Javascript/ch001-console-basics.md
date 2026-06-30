# Console Basics

The browser console is your JavaScript playground. You can open it with **F12** (or **Ctrl+Shift+I**) and click the Console tab.

## Writing to Console

```javascript
console.log("Hello, World!");
console.log(42);
console.log(true);
console.log([1, 2, 3]);
console.log({ name: "Alice", age: 30 });
```

## Different Console Methods

```javascript
console.log("Standard log message");
console.info("Informational message");
console.warn("Warning message");
console.error("Error message");
console.debug("Debug message");
```

## Console Styling

```javascript
console.log("%cStyled text", "color: pink; font-size: 20px; font-weight: bold");
```

## Grouping

```javascript
console.group("User Details");
console.log("Name: Alice");
console.log("Age: 30");
console.groupEnd();
```

## Timing

```javascript
console.time("loop");
for (let i = 0; i < 1000000; i++) {}
console.timeEnd("loop");  // loop: 2.5ms
```

## Creating an Empty HTML Page

```html
<!DOCTYPE html>
<html lang="en-US">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>An empty page</title>
</head>
<body>
    <script src="script.js" defer></script>
</body>
</html>
```

The `<script>` tag loads JavaScript. The `defer` attribute ensures it runs after the HTML is parsed.
