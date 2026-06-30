# String Operations

```javascript
let str = "Hello, JavaScript!";
```

## Properties and Methods

```javascript
str.length;                  // 19
str[0];                      // "H"
str.charAt(0);               // "H"
str.indexOf("Java");         // 7
str.lastIndexOf("a");        // 11
str.includes("Script");      // true
str.startsWith("Hello");     // true
str.endsWith("!");           // true
```

## Extracting Parts

```javascript
str.slice(7, 11);            // "Java" (start, end)
str.slice(7);                // "JavaScript!"
str.slice(-1);               // "!"
str.substring(7, 11);        // "Java" (no negative support)
```

## Case Conversion

```javascript
str.toUpperCase();           // "HELLO, JAVASCRIPT!"
str.toLowerCase();           // "hello, javascript!"
```

## Replacing

```javascript
str.replace("JavaScript", "World");   // "Hello, World!"
str.replace(/a/g, "A");               // "Hello, JAvAScript!" (regex global)
```

## Splitting and Joining

```javascript
"apple,banana,orange".split(",");     // ["apple", "banana", "orange"]
["a", "b", "c"].join(" - ");          // "a - b - c"
```

## Trimming

```javascript
"  hello  ".trim();                   // "hello"
"  hello  ".trimStart();              // "hello  "
"  hello  ".trimEnd();                // "  hello"
```
