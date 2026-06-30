# Filtering with WHERE

## Comparison Operators

| Operation | Operator |
|-----------|----------|
| Equal | `=` |
| Not equal | `<>` |
| Greater than | `>` |
| Greater or equal | `>=` |
| Less than | `<` |
| Less or equal | `<=` |
| String starts with | `STARTS WITH` |
| String ends with | `ENDS WITH` |
| String contains | `CONTAINS` |
| Regular expression | `=~` |

## Basic Filtering

```cypher
MATCH (p:PLAYER)
WHERE p.age > 30
RETURN p
```

## Property Equality (Inline)

If filtering by equality, you can put the condition directly in the node pattern:

```cypher
MATCH (p:PLAYER {age: 33})
RETURN p
```

## Combined Conditions

```cypher
MATCH (p:PLAYER)
WHERE p.weight >= 100 AND p.height <= 2
RETURN p
```

## Logical NOT

```cypher
MATCH (p:PLAYER)
WHERE NOT (p.weight >= 100 AND p.height <= 2)
RETURN p
```

## Calculations in WHERE

```cypher
// Players with BMI > 25
MATCH (p:PLAYER)
WHERE (p.weight / (p.height * p.height)) > 25
RETURN p
```

## String Matching

```cypher
MATCH (c:COACH)
WHERE c.name STARTS WITH 'S'
RETURN c

MATCH (p:PLAYER)
WHERE p.name CONTAINS 'James'
RETURN p

MATCH (t:TEAM)
WHERE t.name ENDS WITH 's'
RETURN t
```
