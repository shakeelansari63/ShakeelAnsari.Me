# Querying Relationships

Relationships are the core of graph databases. They connect nodes and can have their own properties.

## Basic Relationship Query

```cypher
// All players on LA Lakers
MATCH (p:PLAYER) -[r:PLAYS_FOR]-> (t:TEAM)
WHERE t.name = 'LA Lakers'
RETURN p
```

## Relationship Direction

The arrow direction matters. These are equivalent:

```cypher
MATCH (p:PLAYER) -[r:PLAYS_FOR]-> (t:TEAM)
WHERE t.name = 'LA Lakers'
RETURN p

// Same query with reversed arrow
MATCH (t:TEAM) <-[r:PLAYS_FOR]- (p:PLAYER)
WHERE t.name = 'LA Lakers'
RETURN p
```

## Returning Related Nodes

```cypher
MATCH (t:TEAM) <-[r:PLAYS_FOR]- (p:PLAYER)
WHERE t.name = 'LA Lakers'
RETURN p, t
```

## Filtering on Relationship Properties

Relationships can have properties. Filter on them like node properties:

```cypher
// Players earning more than $35M
MATCH (p:PLAYER) -[contract:PLAYS_FOR]-> (t:TEAM)
WHERE contract.salary >= 35000000
RETURN p, t
```

## Multi-Hop Queries

Traverse multiple relationships:

```cypher
MATCH (lebron:PLAYER {name: 'LeBron James'})
    -[:TEAMMATES]-> (p:PLAYER)
MATCH (p) -[contract:PLAYS_FOR]-> (t:TEAM)
WHERE contract.salary >= 35000000
RETURN lebron, p, t
```

## Variable Length Paths

Find nodes that are multiple hops away:

```cypher
// Friends of friends (2 hops)
MATCH (p:PLAYER {name: 'LeBron James'})
    -[:TEAMMATES*2]-> (fof:PLAYER)
RETURN DISTINCT fof
```

## Optional Match

Like `LEFT JOIN` in SQL — returns null for missing matches:

```cypher
MATCH (p:PLAYER)
OPTIONAL MATCH (p) -[r:PLAYS_FOR]-> (t:TEAM)
RETURN p.name, t.name
```
