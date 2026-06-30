# Deleting Nodes and Relationships

## Deleting a Node

```cypher
MATCH (a:SOMETYPE {someproperty: 'some value'})
DELETE a
```

This fails if the node has active relationships — Neo4j prevents dangling relationships.

## Detach Delete

`DETACH DELETE` removes a node and all its relationships:

```cypher
MATCH (ja {name: 'Ja Morant'})
DETACH DELETE ja
```

## Deleting a Relationship

Relationships have no dependencies and can be deleted directly:

```cypher
MATCH (joel {name: 'Joel Embiid'}) -[r:PLAYS_FOR]-> (:TEAM)
DELETE r
```

## Deleting All Nodes and Relationships

```cypher
MATCH (n)
DETACH DELETE n
```

## Removing Properties

Use `REMOVE` to delete specific properties (not the same as deleting a node):

```cypher
MATCH (n:MOVIE {name: "Pushpa, The Rise"})
REMOVE n.release
RETURN n
```

## Practical Cleanup Examples

```cypher
// Delete all players under age 25 and their relationships
MATCH (p:PLAYER)
WHERE p.age < 25
DETACH DELETE p

// Delete all COACHES relationships
MATCH (:COACH) -[r:COACHES]-> (:PLAYER)
DELETE r

// Delete an entire label type
MATCH (n:OLD_LABEL)
DETACH DELETE n
```
