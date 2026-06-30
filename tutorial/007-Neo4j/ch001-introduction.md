# Introduction to Neo4j and Graph Databases

Neo4j is a **graph database** that stores data as nodes, relationships, and properties — unlike relational databases that use tables and foreign keys.

## What is a Graph Database?

A graph database uses graph structures with nodes, edges, and properties to represent and store data.

- **Nodes** — entities (people, places, things)
- **Relationships** — connections between nodes (knows, works_at, bought)
- **Properties** — key-value pairs on nodes and relationships
- **Labels** — categorize nodes (Person, Product, Order)

## Why Graph Databases?

| Feature | Relational DB | Graph DB |
|---------|--------------|----------|
| Data model | Tables, rows, foreign keys | Nodes, relationships |
| Complex joins | Expensive | Natural (traversal) |
| Schema | Rigid, predefined | Flexible |
| Deep relationships | Multiple JOINs | Simple pattern matching |
| Performance | Degrades with depth | Stays constant |

## Cypher Query Language

Neo4j uses **Cypher** — a declarative query language designed for graphs. It's visual and pattern-based.

```cypher
// Match a pattern and return results
MATCH (p:Person) -[:KNOWS]-> (f:Person)
WHERE p.name = 'Alice'
RETURN f.name
```

## Core Concepts

```cypher
// Node with label: PLAYER and properties
(:PLAYER {name: "LeBron James", age: 36, height: 2.06})

// Relationship with type and properties
-[:PLAYS_FOR {salary: 40000000}]->

// Complete pattern
(p:PLAYER) -[r:PLAYS_FOR]-> (t:TEAM)
```

## Use Cases

- Social networks (friends, followers)
- Recommendation engines
- Fraud detection
- Network and IT operations
- Knowledge graphs
- Supply chain management
