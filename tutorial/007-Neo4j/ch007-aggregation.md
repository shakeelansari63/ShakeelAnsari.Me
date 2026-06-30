# Data Aggregation

Cypher provides aggregation functions similar to SQL's `GROUP BY`.

## Aggregation Functions

| Function | Purpose |
|----------|---------|
| `AVG()` | Average |
| `COUNT()` | Count |
| `SUM()` | Sum all values |
| `MAX()` | Maximum value |
| `MIN()` | Minimum value |
| `COLLECT()` | Collect values into a list |
| `STDEV()` | Standard deviation |

## Basic Aggregation

```cypher
MATCH (p:PLAYER) -[game:PLAYED_AGAINST]-> (:TEAM)
RETURN
    p.name,
    COUNT(game) AS games_played,
    AVG(game.points) AS avg_points_per_game
```

## Counting

```cypher
// Count all players
MATCH (p:PLAYER)
RETURN COUNT(p) AS total_players

// Count with filter
MATCH (p:PLAYER)
WHERE p.height > 2.0
RETURN COUNT(p) AS tall_players
```

## Grouping with Multiple Fields

```cypher
MATCH (p:PLAYER) -[r:PLAYS_FOR]-> (t:TEAM)
RETURN
    t.name AS team,
    COUNT(p) AS player_count,
    AVG(r.salary) AS avg_salary,
    MAX(r.salary) AS max_salary,
    MIN(r.salary) AS min_salary
```

## Collect

`COLLECT` creates a list of values:

```cypher
MATCH (t:TEAM) <-[:PLAYS_FOR]- (p:PLAYER)
RETURN t.name AS team, COLLECT(p.name) AS players
```

## Aggregation with Relationships

```cypher
MATCH (p:PLAYER) -[game:PLAYED_AGAINST]-> (:TEAM)
RETURN
    p.name,
    COUNT(game) AS total_games,
    SUM(game.points) AS total_points,
    AVG(game.points) AS avg_points,
    MAX(game.points) AS best_game
ORDER BY avg_points DESC
```
