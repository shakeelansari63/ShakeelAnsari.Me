---
title: Shuffling a Deck Without the Built‑in Random Function
excerpt: I wanted a truly unpredictable shuffle that changes every run, even with the same deck. By hashing a timestamp with SHA‑512 I can generate a stream of pseudo‑random indices and use them to reorder the cards.
date: 2019-08-07
readTime: 5 min read
tags: Python, Algorithms, Cryptography, Randomness
---
![Shuffling a deck](images/shuffle-deck.webp)

So, today I was confronted with a problem which made me think.  
The problem was to build a function to shuffle a deck of cards **without** using any builtin `random` function.  
Also, the deck should be shuffled differently even if the same input is provided.  
So we cannot hardcode any random pattern as that would generate the same output for the same input.

## The challenge

I needed a way to produce **52 pseudo‑random indices** (0–51) that differ every time the program runs.  
The usual `random.randint` is out of the question, so I turned to cryptographic hashing.

### Why hashing?

- A hash takes any input string and outputs a seemingly random fixed‑size string.  
- With a large enough hash (SHA‑512) we get plenty of bits to extract many numbers from.  
- If we change the input string slightly (e.g., by using the current timestamp), the whole hash changes, giving us a different sequence each run.

## My approach

1. **Pick an input string**  
   I used the current timestamp (`time.time()`), converted to a string.  
   Since the timestamp changes each call, the hash will differ.

2. **Hash the string with SHA‑512**  
   ```python
   import hashlib
   h = hashlib.sha512(timestamp_str.encode()).hexdigest()
   ```
   The result is a long hexadecimal string.

3. **Extract numeric pairs**  
   I walked through the hex string two characters at a time, converting each pair to an integer.  
   If the value exceeded 51, I applied `value % 52` to bring it into the 0‑51 range.

4. **Collect 52 numbers**  
   I repeated the extraction until I had exactly 52 indices.  
   Duplicates are fine because the shuffle logic handles them gracefully.

5. **Shuffle the deck**  
   I iterate over the deck, popping a card and inserting it at the position indicated by the next index from the list.  
   If two cards target the same spot, the first one stays, and the second pushes the others to the right.  
   This simple mechanism eliminates the need for a full Fisher‑Yates shuffle.

### Limitations & assumptions

- **Duplicates**: The algorithm permits duplicate indices. In practice this works because the insertion step resolves conflicts, but it’s not the most efficient shuffle.
- **Timestamp precision**: On very fast systems, multiple calls may share the same timestamp, producing identical shuffles. A higher‑resolution timer or additional entropy would mitigate this.
- **Determinism**: The method is deterministic for a given timestamp string. If you need cryptographic‑strength randomness, consider a secure random source instead.

## Code in action

You can see the full implementation on GitHub:

[shuffle_cards.py](https://github.com/shakeelansari63/common_programs/blob/master/shuffle_cards.py)

Feel free to clone, run, and tweak the script. It prints a shuffled deck each time you execute it.

## Final thoughts

Using SHA‑512 to generate a stream of indices is a neat hack when you can’t or don’t want to use Python’s `random` module.  
It’s fast, easy to understand, and—thanks to the timestamp—always produces a different shuffle for a fresh run.  
If you need something more robust, just replace the hash source with a high‑entropy generator and you’re good to go.

Thanks for following along! Until next time, keep those decks spinning.
