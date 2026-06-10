# Installing Go

Go (also called Golang) is a statically typed, compiled programming language.

## Windows

1. Visit [go.dev](https://go.dev/dl/) and download the Windows installer
2. Run the downloaded `.msi` file
3. Follow the installer wizard — it automatically sets up the necessary environment variables
4. Restart your terminal after installation

## Linux

**Using the official tarball (recommended):**
```bash
wget https://go.dev/dl/go1.24.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.24.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc
```

**Using package manager:**
```bash
# Debian/Ubuntu
sudo apt update && sudo apt install golang-go

# Fedora
sudo dnf install golang

# Arch Linux
sudo pacman -S go
```

## macOS

**Using Homebrew (recommended):**
```bash
brew install go
```

**Using the official package:**
Download the macOS installer from [go.dev](https://go.dev/dl/) and run it.

## Verify

```bash
go version
```
