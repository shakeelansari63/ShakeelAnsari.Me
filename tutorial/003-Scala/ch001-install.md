# Installing Scala

Scala runs on the JVM and requires Java to be installed first.

## Install Java

### Windows
Download and run the Java installer from [java.com](https://java.com).

### Linux
```bash
# Debian/Ubuntu
sudo apt update && sudo apt install default-jdk

# Fedora
sudo dnf install java-latest-openjdk

# Arch Linux
sudo pacman -S jre-openjdk
```

### macOS
```bash
brew install java
```

### Verify
```bash
java -version
```

## Install Scala

### Windows
1. Visit [scala-lang.org](https://scala-lang.org/download/) and download the Windows installer
2. Run the installer
3. Alternatively, use coursier:
   - Download coursier from [coursier.io](https://coursier.io)
   - Run: `cs install scala scalac`

### Linux
**Using coursier (recommended):**
```bash
curl -fL https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-pc-linux.gz | gzip -d > cs
chmod +x cs
./cs install scala scalac
```

**Using SDKMAN:**
```bash
curl -s "https://get.sdkman.io" | bash
source "$HOME/.sdkman/bin/sdkman-init.sh"
sdk install scala
```

### macOS
**Using coursier (recommended):**
```bash
brew install coursier/formulas/coursier
cs install scala scalac
```

**Using Homebrew:**
```bash
brew install scala
```

**Using SDKMAN:**
```bash
curl -s "https://get.sdkman.io" | bash
source "$HOME/.sdkman/bin/sdkman-init.sh"
sdk install scala
```

### Verify

```bash
scala -version
scalac -version
```

## Scala REPL

Start the interactive REPL with:

```bash
scala
```

Useful REPL commands:

| Command | Description |
|---------|-------------|
| `:help` or `:he` | Get help about REPL |
| `:quit` or `:q` | Exit REPL |
| `:save` | Save session to file |
| `:replay` | Replay everything in REPL from beginning |
| `:load` | Load and run Scala file |
| `:imports` | See all imports in session |
| `:reset` | Reset REPL session |
| `:sh` | Run shell command from REPL |
| `:silent` | Disable automatic result printing |
| `:paste` | Multi-line input with line breaks |
