# Querying Nodes with MATCH

The `MATCH` keyword finds patterns in the graph. `RETURN` specifies what to output.

## Get All Nodes

```cypher
MATCH (n)
RETURN n
```

## Query by Label

```cypher
MATCH (n:PLAYER)
RETURN n
```

## Query Specific Properties

```cypher
MATCH (n:PLAYER)
RETURN n.name

// Multiple properties
MATCH (n:PLAYER)
RETURN n.name, n.age

// With aliases
MATCH (n:PLAYER)
RETURN n.name AS player_name, n.age AS player_age
```

## Querying Multiple Node Types

```cypher
MATCH (p:PLAYER), (c:COACH)
WHERE p.height >= 2.1 AND c.name STARTS WITH 'S'
RETURN p, c
```

## Ordering Results

```cypher
MATCH (p:PLAYER)
RETURN p
ORDER BY p.height DESC
```

| Order | Keyword |
|-------|---------|
| Ascending | `ASC` |
| Descending | `DESC` |

## Limiting and Skipping

```cypher
// First 3 players
MATCH (p:PLAYER)
RETURN p
LIMIT 3

// Skip 3, show next 3 (pagination)
MATCH (p:PLAYER)
RETURN p
SKIP 3
LIMIT 3
```

## Querying by ID

```cypher
MATCH (n)
WHERE ID(n) = 0
RETURN n
```
