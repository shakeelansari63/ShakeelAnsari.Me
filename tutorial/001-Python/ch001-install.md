# Installing Python

Python is available on all major operating systems. Here's how to install it.

## Windows

1. Visit [python.org](https://python.org) and download the latest Python installer
2. Run the installer
3. **Important:** Check **"Add Python to PATH"** before clicking Install
4. Click Install and wait for the setup to complete

## Linux

Most Linux distributions come with Python pre-installed. To check:

```bash
python3 --version
```

If not installed, use your package manager:

**Debian/Ubuntu:**
```bash
sudo apt update && sudo apt install python3 python3-pip
```

**Fedora:**
```bash
sudo dnf install python3 python3-pip
```

**Arch Linux:**
```bash
sudo pacman -S python python-pip
```

## macOS

macOS no longer ships with Python pre-installed. For Python 3, use one of these methods:

**Using Homebrew (recommended):**
```bash
brew install python3
```

**Using the official installer:**
Download the macOS installer from [python.org](https://python.org) and run it.

## Verify

Open a terminal and run:

```bash
python3 --version
```

You should see the installed version number.
