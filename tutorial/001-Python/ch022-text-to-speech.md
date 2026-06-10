# Text-to-Speech with pyttsx3

`pyttsx3` is a text-to-speech library that works offline. Install with `pip install pyttsx3`.

On Linux, you also need `espeak`:

```bash
sudo apt install espeak
```

## Basic Usage

```python
import pyttsx3

tts = pyttsx3.init()

tts.say('Hello, how are you today?')
tts.runAndWait()
```

## Using espeak on Linux

```python
import pyttsx3

tts = pyttsx3.init('espeak')
tts.say('Hello, how are you today?')
tts.runAndWait()
```

## Voice Properties

```python
tts = pyttsx3.init()

# Get available voices
voices = tts.getProperty('voices')
for voice in voices:
    print(voice.id)

# Set a specific voice
tts.setProperty('voice', voices[0].id)

# Adjust rate and volume
tts.setProperty('rate', 150)       # Speed of speech
tts.setProperty('volume', 0.9)     # Volume (0.0 to 1.0)

tts.say('Hello Shakeel, how are you doing?')
tts.runAndWait()
```

The `runAndWait()` method blocks until the speech is finished.
