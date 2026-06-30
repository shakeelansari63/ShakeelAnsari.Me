# Creating Nodes and Relationships

Use `CREATE` to add nodes and relationships to the database.

## Creating a Single Node

```cypher
CREATE (:MOVIE {name: "YJHD", release: 2013})
```

## Creating a Node with Multiple Labels

```cypher
CREATE (:MOVIE:WEBSERIES {name: "Bahubali"})
```

## Creating and Returning

```cypher
CREATE (bb:MOVIE:WEBSERIES {name: "Bahubali"})
RETURN bb
```

## Creating Multiple Nodes at Once

```cypher
CREATE
    (:ACTOR {name: "Prabhas"}),
    (:ACTOR {name: "Allu Arjun"}),
    (:MOVIE {name: "Pushpa"})
```

## Creating Nodes with Relationships

Create two connected nodes in one statement:

```cypher
CREATE (:ACTOR {name: "Prabhas"})
    -[:ACTED_IN {salary: 20000000}]->
    (:MOVIE {name: "Bahubali"})
```

## Creating Relationships Between Existing Nodes

```cypher
// First create the nodes
CREATE (:MOVIE {name: "Pushpa"}),
       (:ACTOR {name: "Allu Arjun"})

// Then connect them
MATCH (aa:ACTOR {name: "Allu Arjun"}),
      (pushpa:MOVIE {name: "Pushpa"})
CREATE (aa) -[:ACTED_IN]-> (pushpa)
```

## Creating with Multiple Relationships

```cypher
CREATE
    (lebron:PLAYER {name: "LeBron James", age: 36}),
    (lakers:TEAM {name: "LA Lakers"}),
    (frank:COACH {name: "Frank Vogel"}),

    (lebron)-[:PLAYS_FOR {salary: 40000000}]->(lakers),
    (frank)-[:COACHES]->(lebron)
```
