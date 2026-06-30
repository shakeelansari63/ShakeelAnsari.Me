# Updating Data (SET and REMOVE)

## Updating Node Properties with SET

Add new properties or update existing ones:

```cypher
MATCH (p:MOVIE {name: "Pushpa"})
SET p.release = 2020,
    p.name = "Pushpa, The Rise"
RETURN p
```

## Adding Labels

```cypher
MATCH (p:MOVIE {name: "Pushpa, The Rise"})
SET p:HITMOVIE
RETURN p
```

## Updating Relationship Properties

```cypher
// Add salary to an existing ACTED_IN relationship
MATCH (:ACTOR {name: "Allu Arjun"})
    -[worked:ACTED_IN]->
    (:MOVIE {name: "Pushpa, The Rise"})
SET worked.salary = 30000000
RETURN worked
```

## Removing Properties with REMOVE

```cypher
MATCH (n:MOVIE {name: "Pushpa, The Rise"})
REMOVE n.release
RETURN n
```

## Removing Labels

```cypher
MATCH (n:MOVIE {name: "Pushpa, The Rise"})
REMOVE n:HITMOVIE
RETURN n
```

## Updating with Expressions

```cypher
// Give everyone a 10% salary increase
MATCH (p:PLAYER) -[r:PLAYS_FOR]-> (:TEAM)
SET r.salary = r.salary * 1.1
RETURN p.name, r.salary
```

## Merge (Create or Update)

`MERGE` creates a node/relationship if it doesn't exist, or matches it if it does:

```cypher
MERGE (p:PLAYER {name: "Luka Doncic"})
ON CREATE SET p.age = 22, p.height = 2.01
ON MATCH SET p.age = 23
RETURN p
```
