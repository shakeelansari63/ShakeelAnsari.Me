# GORM ORM

[GORM](https://gorm.io/) is a full-featured ORM for Go. It supports migrations, associations, hooks, and multiple database backends (PostgreSQL, MySQL, SQLite, SQL Server).

## Setup

```bash
go get gorm.io/gorm
go get gorm.io/driver/postgres
```

Substitute `gorm.io/driver/mysql` or `gorm.io/driver/sqlite` for other databases.

## Defining Models

Models in GORM are plain Go structs with struct tags for column configuration:

```go
type User struct {
    gorm.Model           // embeds ID, CreatedAt, UpdatedAt, DeletedAt
    Name string `gorm:"size:255"`
}
```

`gorm.Model` is optional but convenient — it adds `ID`, `CreatedAt`, `UpdatedAt`, and `DeletedAt` fields. Without it you must declare a primary key explicitly.

## Connecting to PostgreSQL

```go
import (
    "gorm.io/driver/postgres"
    "gorm.io/gorm"
    "gorm.io/gorm/logger"
)

var DB *gorm.DB

func connectDatabase() {
    dsn := "host=localhost user=postgres password=pass dbname=postgres port=5432 sslmode=disable"
    database, err := gorm.Open(postgres.Open(dsn), &gorm.Config{
        Logger: logger.Default.LogMode(logger.Info),
    })
    if err != nil {
        panic("Failed to connect to database")
    }
    DB = database
}
```

The DSN string contains connection parameters. GORM uses a driver interface so you can swap databases by changing the import and Open call.

## AutoMigrate

```go
DB.AutoMigrate(&User{})
```

`AutoMigrate` creates or alters tables to match the struct definition. It is safe to call on every startup — it only adds columns, never drops them.

## CRUD Operations

**Create:**

```go
user := User{ID: 1, Name: "John"}
DB.Create(&user)
```

`Create` inserts the record and sets the primary key on the passed pointer.

**Query (multiple rows):**

```go
var users []User
DB.Where("name LIKE ?", "J%").Find(&users)
```

`Find` populates the slice with all matching rows. The `?` placeholder prevents SQL injection.

**Query (single row):**

```go
var user User
DB.Where("id = ?", 2).First(&user)
```

`First` returns the first matching record or `ErrRecordNotFound`.

**Update:**

```go
user.Name = "James C"
DB.Save(&user)
```

`Save` updates all fields of an existing record. Use `DB.Model(&user).Update("name", "James C")` to update a single column.

**Delete:**

```go
DB.Where("1 = 1").Unscoped().Delete(&User{})
```

By default GORM uses soft deletes — it sets `deleted_at` instead of removing the row. `.Unscoped()` bypasses this and performs a hard delete.

## Logger Configuration

```go
newLogger := logger.New(
    log.New(os.Stdout, "\n", log.LstdFlags),
    logger.Config{
        SlowThreshold:             time.Second,
        LogLevel:                  logger.Info,
        IgnoreRecordNotFoundError: true,
        Colorful:                  true,
    },
)
```

A custom logger lets you set a slow-query threshold, control verbosity, and format output. GORM logs every SQL statement when `LogLevel` is `logger.Info`.
