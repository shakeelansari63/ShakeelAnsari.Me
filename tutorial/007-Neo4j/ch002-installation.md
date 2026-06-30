# Installing Neo4j

Neo4j is available as a desktop application, Docker container, or cloud service.

## Using Docker (Recommended)

```bash
docker run -d \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:5
```

- `7474` — Neo4j Browser (web UI)
- `7687` — Bolt protocol (driver connections)

## Neo4j Desktop

Download from [neo4j.com/download](https://neo4j.com/download) — includes the browser, database management, and project management.

## Loading Initial Data

```cypher
CREATE
(russell:PLAYER{name:"Russell Westbrook", age: 33, number: 0, height: 1.91, weight: 91}),
(lebron:PLAYER{name:"LeBron James", age: 36, number: 6, height: 2.06, weight: 113}),
(anthony:PLAYER{name:"Anthony Davis", age: 28, number: 23, height: 2.08, weight: 115}),
(lakers:TEAM{name:"LA Lakers"}),
(memphis:TEAM{name:"Memphis Grizzlies"}),

(lebron)-[:PLAYS_FOR {salary: 40000000}]-> (lakers),
(russell)-[:PLAYS_FOR {salary: 33000000}]-> (lakers),
(anthony)-[:PLAYS_FOR {salary: 38000000}]-> (lakers)
```

## Connecting with a Driver

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

with driver.session() as session:
    result = session.run("MATCH (n) RETURN n")
    for record in result:
        print(record)
```

## Neo4j Browser

Open `http://localhost:7474` in your browser to access the Neo4j Browser — an interactive query editor with graph visualization.
