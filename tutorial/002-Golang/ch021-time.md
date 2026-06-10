# Time

Go has a full-featured `time` package, but its formatting approach is unique and famously different from other languages.

## The Reference Layout

Most languages use format codes like `YYYY-MM-DD`. Go instead uses a **reference date**:

```
Mon Jan 2 15:04:05 MST 2006
```

The mnemonic is `01 02 03 04 05 06 07`:

| Number | Meaning      | Reference Value |
|--------|-------------|-----------------|
| `01`   | Month       | January / Jan   |
| `02`   | Day         | 2nd / 02        |
| `03`   | Hour (12h)  | 03 / 15         |
| `04`   | Minute      | 04              |
| `05`   | Second      | 05              |
| `06`   | Year        | 2006            |
| `-07`  | Timezone    | -0700 / MST     |

2nd January 2006 was a Monday, so `Mon` or `Monday` also work for day names.

```go
currentTime := time.Now()
fmt.Println(currentTime.Format("02/01/2006"))
fmt.Println(currentTime.Format("Mon 02 January 2006"))
fmt.Println(currentTime.Format("2006-01-02 15:04:05"))
```

## Constructing Dates

Use `time.Date` to create a specific time:

```go
newTime := time.Date(2022, time.June, 5, 0, 0, 0, 0, time.Local)
fmt.Println(newTime.Format("Mon 02 January 2006"))
```

The parameters are year, month, day, hour, minute, second, nanosecond, and location. Month names like `time.June` are preferred over numeric values.
