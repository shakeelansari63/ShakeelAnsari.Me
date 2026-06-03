---
title: Shuffling a Deck Without Built‑in Random  
excerpt: A step‑by‑step look at a Python function that uses timestamps and SHA‑512 to generate a shuffled deck without the random module.  
date: 2019-08-07  
readTime: 8 min read  
tags: Python, Algorithms, Randomness, Shuffling, Hashing  
---

![Shuffling a deck](images/shuffle-deck.webp)

## Hey there, fellow card‑lover!

I ran into a quirky challenge the other day: **shuffle a standard 52‑card deck without calling any of Python’s built‑in random functions**. The twist? The shuffle should look different each time, even when the same ordered deck is fed in. I whipped up a small script and thought I’d walk you through how it works, why I made each decision, and where the approach might fall short.

## The problem, plain and simple  

We need a function `shuffle_cards(deck)` that:

1. Accepts a list of 52 distinct card identifiers.  
2. Rearranges the list into a new order that changes on every execution.  
3. Avoids `random`, `numpy.random`, or any other library that directly supplies randomness.

## My overall strategy  

Because we can’t call a random generator, I turned to **deterministic data that changes on its own** – the current timestamp. By hashing that timestamp with SHA‑512 I obtain a long, seemingly unpredictable hexadecimal string. From that string I extract digits, massage them into indices 0‑51, and finally use those indices to reposition the cards.

## Let me take you through the code  

### 1. Seed generation

```python
seed = datetime.strftime(datetime.today(), '%Y%m%d%H%M%S%f')
```

- **What it does:** Formats the current date‑time down to microseconds into a string like `20260603142530123456`.  
- **Reasoning:** Every program run (even within the same second) will produce a different seed because of the microsecond part, giving us a source of variability without calling a random function.

### 2. Hashing the seed  

```python
hash_seed = sha512(seed.encode('ascii')).hexdigest()
```

- **What it does:** Turns the seed into a 128‑character hexadecimal digest.  
- **Reasoning:** Cryptographic hashes spread input bits uniformly across the output, turning a predictable timestamp into a more “random‑looking” pattern. SHA‑512 is overkill for a simple shuffle, but it guarantees a long string to draw numbers from.

### 3. Extracting digits only  

```python
number_str = "".join(re.compile(r'[0-9]').findall(hash_seed))
```

- **What it does:** Strips out any letters (`a‑f`) and leaves only the decimal digits.  
- **Reasoning:** We need numeric values to serve as list indices. Using only digits keeps the conversion to integers straightforward.

### 4. Building 52 position numbers  

The first `while` loop grabs **adjacent digit pairs** (`34`, `78`, …). If a pair exceeds `51`, we wrap it with modulo 52.

- **Why two‑digit pairs?** A pair gives us a range of 00‑99, which covers the required 0‑51 comfortably while still providing many possible values.
- **Modulo handling:** Guarantees every number lands inside the valid index range.

If the first pass doesn’t yield 52 numbers (possible when the digit string is short), the second loop switches to a **skip‑one pattern** (`0` with `2`, `1` with `3`, …) to squeeze more values out of the same string.

### 5. Reordering the deck  

```python
for old_pos, new_pos in enumerate(new_pos):
    card = deck.pop(old_pos)
    deck.insert(new_pos, card)
```

- **What it does:** Iterates over the generated index list, removes the card at the original position, and inserts it at the new position.
- **Why `pop` + `insert`?** This directly mutates the original list, matching the “in‑place shuffle” requirement without extra containers.

## Limitations & assumptions  

**Not cryptographically secure.** An attacker who knows the exact timestamp could reproduce the shuffle.  
  
**Deterministic order of operations.** The algorithm depends on the order of `enumerate(new_pos)`. Changing that order would change the final deck.  
  
**Microsecond resolution.** If the environment’s clock only offers second precision, consecutive runs may produce identical shuffles.  
  
If you need a truly unbiased, secure shuffle, a proper random number generator (e.g., `random.shuffle` or `secrets.randbelow`) would be the right tool. This method is mainly a fun exercise in “how else can we get variation?”

## Running the demo  

Save the script as `shuffle_deck.py` and execute:

```bash
python shuffle_deck.py
```

Each run prints a different ordering of the 52 cards. Feel free to copy the output into a text file and compare runs - you’ll see the variation in action.

## Wrapping up  

I enjoyed turning a timestamp into a makeshift randomness source, hashing it, and then coaxing those bits into deck positions. The code is short, self‑contained, and demonstrates how deterministic data can be repurposed when built‑in randomness isn't an option. Just keep the limitations in mind if you ever need stronger guarantees.
  
#### Happy shuffling
