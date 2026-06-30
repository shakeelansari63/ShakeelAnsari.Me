# Pandas — Series & DataFrame

Pandas provides `Series` and `DataFrame` for working with tabular data. Install with `pip install pandas`.

```python
import pandas as pd
```

## Series

A Series is like a key-value pair (index + values).

```python
series = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
print(series)
# a    10
# b    20
# c    30
# d    40
# e    50

print(series['c'])       # 30 — access by label
print(series.iloc[2])    # 30 — access by integer index

series.name = "My Series"
print(dict(series))      # {'a': 10, 'b': 20, 'c': 30, 'd': 40, 'e': 50}
```

### Series Operations

```python
s1 = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
s2 = pd.Series([100, 90, 80, 60, 70], index=['c', 'b', 'a', 'd', 'e'])

print(s1 + s2)        # Sum by matching index
print(s1.head(3))     # First 3 items
print(s1.tail(3))     # Last 3 items
print(s1.count())     # Number of items
```

### Series Manipulation

```python
s1 = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
s2 = pd.Series([100, 90, 80, 60, 70], index=['c', 'b', 'a', 'd', 'e'])

print(s1.apply(lambda x: x**2))    # Apply function to all elements
print(s2.sort_index())             # Sort by index
print(s2.sort_values())            # Sort by value

s2.sort_index(inplace=True)        # Modify in-place
```

## DataFrames

A DataFrame is a table with rows and columns, like a spreadsheet.

```python
data = {
    'name': ['Anna', 'Bob', 'Mike'],
    'age': [21, 42, 39],
    'height': [157, 178, 145],
    'gender': ['f', 'm', 'm']
}

df = pd.DataFrame(data)
print(df)
#    name  age  height gender
# 0  Anna   21     157      f
# 1   Bob   42     178      m
# 2  Mike   39     145      m
```

### Custom Index

```python
df.set_index('name', inplace=True)
print(df)
#        age  height gender
# name
# Anna    21     157      f
# Bob     42     178      m
# Mike    39     145      m
```

### Basic Attributes

```python
print(df.head(2))    # First 2 rows
print(df.tail(2))    # Last 2 rows
print(df.shape)      # (rows, columns)
print(df.dtypes)     # Types of each column
print(df.T)          # Transpose
```

### Accessing Data

```python
print(df.iloc[1])         # 2nd row
print(df.index)           # Index column (name)
print(df.index[1])        # Name in 2nd row
```

### Reading CSV

```python
df = pd.read_csv('people.csv', delimiter=',')
```

### Statistical Functions

```python
print(df.count())       # Count of non-null values
print(df.sum())         # Sum of each column
print(df['height'].mean())    # Average
print(df['height'].median())  # Median
print(df['height'].std())     # Standard deviation
print(df['height'].describe()) # Summary statistics (count, mean, std, min, 25%, 50%, 75%, max)
```

### Sorting

```python
df.sort_index(inplace=True)
df.sort_values(by=['height', 'age'], inplace=True)
```

### Filtering

```python
# Filter rows where age > 35
my_cond = df['age'] > 35
print(df.loc[my_cond])

# Multiple conditions
print(df.loc[(df['height'] > 160) & (df['age'] < 35)])
```

### Merging DataFrames

```python
inner = pd.merge(df1, df2, on='id', how='inner')
left  = pd.merge(df1, df2, on='id', how='left')
right = pd.merge(df1, df2, on='id', how='right')
outer = pd.merge(df1, df2, on='id', how='outer')
```

### Reading Excel Files

```python
df = pd.read_excel('datasets/lures.xlsx', sheet_name='LURES')
```

### Grouping and Aggregation

```python
# Group by a column and compute aggregates
df.groupby('gender')['height'].mean()
df.groupby('gender')['age'].agg(['mean', 'std', 'count'])
```

### Built-in Plotting

DataFrames include a `.plot()` method (wraps Matplotlib):

```python
df.plot(kind='scatter', x='height', y='age')
plt.show()

df.plot(kind='bar', x='name', y='age')
plt.show()
```
