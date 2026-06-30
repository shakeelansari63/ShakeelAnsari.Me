# Maven Build Tool

Maven is a build automation tool for Java projects. It follows the principle of **convention over configuration**.

## Project Structure

Maven enforces a standard directory layout:

```
my-project/
├── pom.xml
├── src/
│   ├── main/
│   │   ├── java/        # source code
│   │   ├── resources/   # config files, properties
│   │   └── webapp/      # web application files (WAR)
│   └── test/
│       ├── java/        # test source code
│       └── resources/   # test resources
└── target/              # compiled output
```

## POM (Project Object Model)

`pom.xml` is the heart of a Maven project. It contains metadata and configuration for building the project.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project
    xmlns="http://maven.apache.org/POM/4.0.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
        http://maven.apache.org/xsd/maven-4.0.0.xsd">

    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>my-project</artifactId>
    <version>1.0-SNAPSHOT</version>
    <packaging>jar</packaging>
</project>
```

## Common Maven Commands

| Command | Purpose |
|---------|---------|
| `mvn clean` | Delete the `target/` directory |
| `mvn compile` | Compile source code to `target/classes/` |
| `mvn test` | Run unit tests |
| `mvn package` | Create JAR/WAR in `target/` |
| `mvn install` | Install JAR into local repository |
| `mvn deploy` | Deploy to remote repository |

## Dependencies

```xml
<dependencies>
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-core</artifactId>
        <version>5.3.20</version>
    </dependency>
</dependencies>
```

## Dependency Scopes

| Scope | Available During | Included in Final |
|-------|-----------------|-------------------|
| `compile` (default) | Compilation + Runtime | Yes |
| `provided` | Compilation only | No (provided by runtime) |
| `runtime` | Runtime only | Yes |
| `test` | Test only | No |
| `system` | Compilation only | No (must be on classpath) |

## Plugins

```xml
<build>
    <plugins>
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-compiler-plugin</artifactId>
            <version>3.8.1</version>
            <configuration>
                <release>17</release>
            </configuration>
        </plugin>
    </plugins>
</build>
```

## Build Lifecycle Phases

Maven executes phases in order:

1. **validate** — validate project structure
2. **compile** — compile source code
3. **test** — run tests
4. **package** — package into JAR/WAR
5. **verify** — run integration tests
6. **install** — add to local repository
7. **deploy** — upload to remote repository

Running `mvn install` executes all phases up to and including `install`.
