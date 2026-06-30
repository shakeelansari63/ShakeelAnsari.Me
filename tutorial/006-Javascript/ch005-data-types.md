# Data Types

JavaScript has 7 primitive data types and 1 complex type (Object).

## Primitive Types

| Type | Example | Description |
|------|---------|-------------|
| `number` | `42`, `3.14`, `NaN` | Integers and floats |
| `string` | `"hello"`, `'world'` | Text |
| `boolean` | `true`, `false` | Logical values |
| `undefined` | `undefined` | Variable declared but not assigned |
| `null` | `null` | Intentional absence of value |
| `bigint` | `9007199254740991n` | Large integers |
| `symbol` | `Symbol("id")` | Unique identifier |

## Checking Types

```javascript
typeof 42;              // "number"
typeof "hello";         // "string"
typeof true;            // "boolean"
typeof undefined;       // "undefined"
typeof null;            // "object" (this is a known JS bug)
typeof 9007199254740991n; // "bigint"
typeof Symbol();        // "symbol"
typeof {};              // "object"
typeof [];              // "object"
```

## Type Coercion

JavaScript automatically converts types when operators are applied:

```javascript
"5" + 3;      // "53" (number coerced to string)
"5" - 3;      // 2   (string coerced to number)
"5" * "2";    // 10  (both strings coerced to numbers)
"hello" - 1;  // NaN  (Not a Number)
```

## Explicit Type Conversion

```javascript
Number("42");           // 42
String(42);             // "42"
Boolean(1);             // true
parseInt("42px");       // 42
parseFloat("3.14em");   // 3.14
```

## Falsy Values

These values evaluate to `false` in boolean contexts:

- `false`
- `0`, `-0`
- `""` (empty string)
- `null`
- `undefined`
- `NaN`

Everything else is truthy, including `"0"`, `"false"`, `[]`, and `{}`.
