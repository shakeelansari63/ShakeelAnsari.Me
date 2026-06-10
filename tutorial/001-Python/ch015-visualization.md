# Data Visualization — Matplotlib, Seaborn & Plotly

Matplotlib creates plots. Seaborn (built on Matplotlib) simplifies statistical plots. Plotly creates interactive charts.

---

## Matplotlib Basics

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

### Line Plot

```python
x = np.arange(-100, 101, 1)
y = x**2 + 2*x + 4

plt.plot(x, y)
plt.show()
```

### Multiple Lines in One Plot

```python
x = np.arange(-100, 101, 1)
y1 = x**2 + 2*x + 4
y2 = 2000 * np.sin(x)
y3 = 2000 * np.log(abs(x))

plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.show()
```

### Styling Lines

The third argument to `plot()` sets color and line style.

| Pattern | Meaning                |
|---------|------------------------|
| `'r-'`  | Red solid line         |
| `'r--'` | Red dashed line        |
| `'r-.'` | Red dash-dot line      |
| `'ro'`  | Red dotted markers     |

```python
plt.plot(x, y1, 'r--')
plt.plot(x, y2, 'g-.')
plt.plot(x, y3, 'b.')
```

### Labels and Titles

```python
plt.title("Sine Function")
plt.xlabel("Time")
plt.ylabel("Amplitude")
```

### Legends

```python
plt.plot(x, y1, 'r--', label="Sine")
plt.plot(x, y2, 'y-', label="Cosine")
plt.legend(loc='upper right')
```

## Bar Charts

```python
data = [20, 56, 71, 92]
idx = np.arange(0, 4, 1)
plt.bar(idx, data)
plt.show()
```

Grouped bar charts:

```python
idx = np.arange(0, 8, 2)
plt.bar(idx, skill['python'], width=0.2, label="Python")
plt.bar(idx+0.2, skill['java'], width=0.2, label="Java")
plt.xticks(idx+0.3, people)
plt.legend()
```

## Pie Charts

```python
data = [20, 56, 71, 92]
labels = ['Grade A', 'Grade D', 'Grade C', 'Grade B']
plt.pie(data, labels=labels)
plt.show()
```

With customization:

```python
plt.pie(data, labels=labels, autopct='%.2f%%', explode=[0.1, 0.01, 0.01, 0.01],
        startangle=90, counterclock=False)
plt.title("Students Grades")
```

## Subplots

Subplots place multiple plots in one figure. `subplot(rows, cols, index)`.

```python
x = np.arange(0, 101, 0.1)
y1 = 4*(x**2) + 4
y2 = np.sin(x)

sp1 = plt.subplot(211)   # 2 rows, 1 column, 1st plot
sp2 = plt.subplot(212)   # 2 rows, 1 column, 2nd plot

sp1.plot(x, y1)
sp2.plot(x, y2)

plt.tight_layout()
plt.show()
```

### Multiple Figures

Use `plt.figure()` to create separate windows for different plots.

```python
x = np.arange(-10, 11, 0.1)
y1 = 4*(x**2) + 4
y2 = np.sin(x)
y3 = np.log(x)
y4 = x**3

plt.figure(1)
sp1 = plt.subplot(211)
sp2 = plt.subplot(212)
sp1.plot(x, y1, 'r-')
sp2.plot(x, y2, 'g-.')
plt.tight_layout()

plt.figure(2)
plt.plot(x, y3, 'b.')

plt.figure(3)
plt.plot(x, y4, 'y--')

plt.show()
```

### Customizing with Styles

```python
from matplotlib import style as sy

sy.use('ggplot')              # Predefined style

sy.use('default')
plt.grid(True)
```

## Histograms

Histograms show the distribution of data.

```python
data = pd.read_csv('datasets/countries.csv')
data_2007 = data[data.year == 2007]
asia_2007 = data_2007[data_2007.continent == 'Asia']
europe_2007 = data_2007[data_2007.continent == 'Europe']

plt.hist(asia_2007.gdpPerCapita, bins=20, alpha=0.5, label='Asia')
plt.hist(europe_2007.gdpPerCapita, bins=20, alpha=0.5, label='Europe')
plt.legend()
plt.show()
```

### Advanced Histogram Options

```python
rng = np.random.default_rng(seed=112)
data_rand = rng.normal(loc=150, scale=10, size=500)

# Density (probability density instead of counts)
plt.hist(data_rand, density=True)
plt.title("Probability Density")
plt.show()

# Cumulative histogram
plt.hist(data_rand, cumulative=True)
plt.title("Cumulative Histogram")
plt.show()

# Cumulative probability density
plt.hist(data_rand, density=True, cumulative=True)
plt.title("Cumulative Probability Density")
plt.show()

# Step outline (no fill)
plt.hist(data_rand, histtype='step')
plt.title("Stepped Histogram")
plt.show()

# Adjust bar width (gap between bars)
plt.hist(data_rand, rwidth=0.8)
plt.title("Bar Width = 0.8")
plt.show()

# Specify bin count manually
plt.hist(data_rand, bins=15)
plt.title("15 Bins")
plt.show()

# Auto bin method
plt.hist(data_rand, bins='sqrt')
plt.title("SQRT Bins")
plt.show()
```

### Histogram `range` for Consistent Axes

When comparing distributions side-by-side, use `range` to unify the X-axis:

```python
# Without range — each subplot auto-scales
plt.subplot(121)
plt.hist(asia_2007.gdpPerCapita, bins=20)
plt.subplot(122)
plt.hist(europe_2007.gdpPerCapita, bins=20)
plt.show()

# With range — same X-axis scale
plt.subplot(121)
plt.hist(asia_2007.gdpPerCapita, bins=20, range=(0, 50000))
plt.subplot(122)
plt.hist(europe_2007.gdpPerCapita, bins=20, range=(0, 50000))
plt.show()
```

## Scatter Plots

Scatter plots show the relationship between two variables.

```python
data = pd.read_csv('datasets/countries.csv')
data_2007 = data[data.year == 2007]

plt.scatter(data_2007.gdpPerCapita, data_2007.lifeExpectancy)
plt.show()
```

Use `np.log10()` on skewed data to reveal linear relationships:

```python
plt.scatter(np.log10(data_2007.gdpPerCapita), data_2007.lifeExpectancy, s=5)
plt.show()
```

### Multi-panel Scatter with Subplots

```python
import math
years = sorted(set(data.year))
columns = 3
rows = math.ceil(len(years) / columns)

plt.figure(figsize=(12, 8))
for i, year in enumerate(years):
    plt.subplot(rows, columns, i + 1)
    plt.scatter(data[data.year == year].gdpPerCapita,
                data[data.year == year].lifeExpectancy, s=2)
    plt.ylim(25, 90)
    plt.title(f'Year: {year}')

plt.suptitle("GDP per Capita vs Life Expectancy")
plt.tight_layout()
plt.show()
```

## Box Plots

Box plots show data distribution summary: median, quartiles, whiskers and outliers.

```python
# Basic box plot
plt.boxplot(data_rand)
plt.title("Box Plot")
plt.show()

# Show mean marker (triangle)
plt.boxplot(data_rand, showmeans=True)
plt.title("With Mean Marker")
plt.show()

# Show mean as a line instead of marker
plt.boxplot(data_rand, showmeans=True, meanline=True)
plt.title("Mean Line")
plt.show()

# Notch display — useful when comparing multiple boxes
data3 = rng.integers(low=1, high=10, size=(50, 3))
plt.boxplot(data3, tick_labels=['A', 'B', 'C'], notch=True)
plt.title("Notched Box Plot")
plt.show()

# Horizontal box plot
plt.boxplot(data_rand, vert=False)
plt.title("Horizontal Box Plot")
plt.show()
```

## Violin Plots

Violin plots combine box plot summary with kernel density estimation for a richer view of data distribution.

```python
plt.violinplot(data_rand, showmeans=True, showmedians=True)
plt.title("Violin Plot")
plt.show()
```

### Comparing Histogram, Box Plot and Violin Plot

```python
plt.figure(figsize=(12, 6))

plt.subplot(221)
plt.hist(data_rand)

plt.subplot(223)
plt.boxplot(data_rand, showmeans=True, meanline=True, vert=False)

plt.subplot(122)
plt.violinplot(data_rand, showmeans=True, showmedians=True)

plt.tight_layout()
plt.show()
```

### Bimodal Data — Where Violin Plots Shine

Violin plots reveal multi-modal distributions that box plots hide.

```python
d1 = rng.normal(loc=150, scale=10, size=500)
d2 = rng.normal(loc=230, scale=5, size=500)
data4 = np.concatenate([d1, d2])

plt.figure(figsize=(12, 6))

plt.subplot(221)
plt.hist(data4, bins=25)

plt.subplot(223)
plt.boxplot(data4, showmeans=True, meanline=True, vert=False)

plt.subplot(122)
plt.violinplot(data4, showmeans=True, showmedians=True)

plt.tight_layout()
plt.show()
```

Histogram shows bimodality but not summary statistics. Box plot shows summary but not modality. Violin plot shows both.

## Dual Y-Axis with `twinx`

Use `ax.twinx()` when datasets share an X-axis but need different Y-scales.

```python
data = pd.read_csv('datasets/countries.csv')
top10_2007 = data[data.year == 2007].sort_values('population', ascending=False).head(10)

x = np.arange(10)
fig, ax = plt.subplots()

width = 0.3
plt.xticks(x, top10_2007.country, rotation=45)

# Population on left Y-axis
population = ax.bar(x, top10_2007.population / 1e6, width, label="Population")
ax.set_ylabel("Population (millions)")

# GDP on right Y-axis
ax2 = ax.twinx()
gdp = ax2.bar(x + width, (top10_2007.population * top10_2007.gdpPerCapita) / 1e9,
              width, color='orange', label="GDP")
ax2.set_ylabel("GDP (billions)")

plt.legend([population, gdp], ["Population", "GDP"])
plt.title("Top 10 Countries by Population and GDP")
plt.show()
```

## Pandas Built-in Plotting

Pandas DataFrames have a built-in `.plot()` method that wraps Matplotlib.

```python
df = pd.read_csv('datasets/countries.csv')
us_data = df[df.country == 'United States']

# Line plot from DataFrame columns
plt.plot(us_data.year, us_data.gdpPerCapita)
plt.title("US GDP per Capita over Time")
plt.show()

# Bar chart
top10_2007 = data[data.year == 2007].sort_values('population', ascending=False).head(10)
top10_2007.plot(kind='bar', x='country', y='population')
plt.show()

# Scatter via pandas
df_2007 = df[df.year == 2007]
df_2007.plot(kind='scatter', x='gdpPerCapita', y='lifeExpectancy')
plt.show()

# Growth normalisation (compare growth from first year)
us_data['gdp_growth'] = us_data.gdpPerCapita / us_data.gdpPerCapita.iloc[0]
plt.plot(us_data.year, us_data.gdp_growth)
plt.title("US GDP Growth (normalized to 1952)")
plt.show()
```

## Advanced Subplot Layout with `subplot_mosaic`

`subplot_mosaic()` provides more flexible grid layouts than `subplot()`.

```python
fig = plt.figure(tight_layout=True)
axes = fig.subplot_mosaic([['box', 'violin'],
                           ['hist', 'violin'],
                           ['hist', 'violin']])

axes['hist'].hist(data4, bins=25)
axes['box'].boxplot(data4, showmeans=True, meanline=True, vert=False)
axes['violin'].violinplot(data4, showmeans=True, showmedians=True)
plt.show()
```

---

## Seaborn — Statistical Data Visualization

Seaborn is built on Matplotlib. It makes complex statistical plots easy.

```python
import seaborn as sns
```

### Relationship Plots (relplot)

```python
countries = pd.read_csv('datasets/countries.csv')
sns.relplot(data=countries, x='population', y='gdpPerCapita')
plt.show()
```

### Univariate Plots

```python
data = pd.read_csv('datasets/winequality-white.csv', sep=r'\s*;\s*', engine='python')

# Histogram with KDE
sns.displot(data=data.alcohol, kde=True, height=5, aspect=3)
plt.show()

# Add rug (individual data points)
sns.displot(data=data.alcohol, kde=True, rug=True, height=5, aspect=3)
plt.show()

# Rug plot only
sns.rugplot(data=data.alcohol, height=0.5)
plt.show()

# KDE without histogram
sns.kdeplot(data=data.alcohol)
plt.show()

# KDE as fill curve
sns.displot(data=data.alcohol, kind='kde', fill=True, aspect=3)
plt.show()
```

### Categorical Plots with `catplot`

```python
tips = sns.load_dataset('tips')

# Swarm plot (non-overlapping points, categorical X)
sns.catplot(data=tips, x='day', y='tip', kind='swarm')
plt.show()

# Box plot by category
sns.catplot(data=tips, x='day', y='tip', kind='box', hue='sex')
plt.show()

# Violin plot by category
sns.catplot(data=tips, x='day', y='tip', kind='violin', hue='sex')
plt.show()
```

### Bivariate Plots

```python
# Scatter plot with regression line
sns.lmplot(data=countries, x='population', y='gdpPerCapita')
plt.show()

# lmplot with hue, col, row
sns.lmplot(data=countries, x='population', y='gdpPerCapita',
           hue='continent', col='year', col_wrap=3)
plt.show()

# Joint plot (hexbin)
sns.jointplot(data=countries, x='population', y='gdpPerCapita', kind='hex')
plt.show()

# Joint plot (scatter + KDE)
sns.jointplot(data=countries, x='population', y='gdpPerCapita', kind='kde')
plt.show()

# Pair plot — matrix of scatter plots for all numeric columns
iris = sns.load_dataset('iris')
sns.pairplot(iris, hue='species', diag_kind='kde')
plt.show()

# Heatmap — visualise correlation matrix
corr = iris.drop(columns='species').corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.show()
```

### Multivariate Plots with Trellis

Use `col`, `row`, or `hue` parameters to split data across subplots.

```python
sns.relplot(data=countries, x='population', y='gdpPerCapita',
            hue='continent', col='year')
plt.show()
```

#### FacetGrid — Manual Trellis Control

```python
g = sns.FacetGrid(countries, col='continent', hue='continent', col_wrap=3)
g.map(plt.scatter, 'gdpPerCapita', 'lifeExpectancy')
g.set_axis_labels('GDP per Capita', 'Life Expectancy')
g.add_legend()
plt.show()
```

#### PairGrid — Custom Pair Plot

```python
g = sns.PairGrid(iris, hue='species')
g.map_offdiag(plt.scatter)
g.map_diag(plt.hist)
g.add_legend()
plt.show()

# Different plots for upper/lower triangle
g = sns.PairGrid(iris, hue='species')
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
g.map_diag(sns.histplot)
g.add_legend()
plt.show()
```

### Seaborn Customization

```python
# Available styles: 'dark', 'darkgrid', 'white', 'whitegrid', 'ticks'
sns.set_style('darkgrid')

# Remove top/right spines
sns.despine()

# Set context for larger fonts ('paper', 'notebook', 'talk', 'poster')
sns.set_context('poster')

# Color palette types
#   Qualitative: 'pastel', 'husl', 'Set1', 'Set2'
#   Sequential:  'Blues', 'Greens', 'Reds'
#   Diverging:   'coolwarm', 'RdBu', 'Spectral'
sns.set_palette('husl')

# See all available colours
print(sns.color_palette())

# Override individual axes style
sns.axes_style()
```

---

## Plotly — Interactive Charts

Plotly creates interactive plots that can be viewed in browsers. Two modes: online (requires account) and offline (local HTML).

```python
import plotly.offline as pyo
import plotly.graph_objs as go

# Simple scatter plot
trace = go.Scatter(x=[1, 2, 3, 4], y=[1, 4, 9, 16], mode='lines+markers')
data = [trace]
pyo.plot(data, filename='simple-plot.html')
```

### Plotly Layout

```python
layout = go.Layout(title='My Plot', xaxis=dict(title='X'),
                   yaxis=dict(title='Y'))
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)
```

### Annotations and Line Styling

```python
trace = go.Scatter(
    x=[1, 2, 3, 4],
    y=[1, 4, 9, 16],
    mode='lines+markers',
    line=dict(color='rgb(0, 250, 24)', width=4, dash='dash')
)

layout = go.Layout(
    title='Styled Plot',
    annotations=[
        dict(x=3, y=9, xref='x', yref='y',
             text='Peak', showarrow=True, ax=20, ay=-30)
    ]
)

fig = go.Figure(data=[trace], layout=layout)
pyo.plot(fig)
```

### Basic Plots

```python
# Bar chart
trace = go.Bar(x=['A', 'B', 'C'], y=[10, 20, 15])
pyo.plot([trace])

# Pie chart
trace = go.Pie(labels=['Python', 'Java', 'C++'], values=[40, 30, 30])
pyo.plot([trace])
```

### Intermediate Statistical Charts

```python
# Box plot
trace = go.Box(y=data['alcohol'], name='Alcohol')
pyo.plot([trace])

# Histogram
trace = go.Histogram(x=data['alcohol'], nbinsx=20)
pyo.plot([trace])

# Heatmap
correlation_matrix = iris.drop(columns='species').corr()
trace = go.Heatmap(z=correlation_matrix.values,
                   x=correlation_matrix.columns,
                   y=correlation_matrix.columns)
pyo.plot([trace])
```

### Advanced Charts: Bubble, Candlestick, Funnel

Bubble charts encode a third dimension via marker size.

```python
trace = go.Scatter(
    x=[1, 2, 3, 4],
    y=[10, 20, 15, 25],
    mode='markers',
    marker=dict(
        size=[20, 40, 60, 30],
        sizeref=2,
        sizemode='area',
        colorscale='Viridis',
        showscale=True
    )
)
pyo.plot([trace])
```

Candlestick charts for financial time series:

```python
trace = go.Candlestick(
    x=['2023-01-01', '2023-01-02', '2023-01-03'],
    open=[100, 105, 102],
    high=[110, 108, 107],
    low=[95, 100, 98],
    close=[105, 102, 106]
)
layout = go.Layout(xaxis=dict(rangeslider=dict(visible=True), type='date'))
fig = go.Figure(data=[trace], layout=layout)
pyo.plot(fig)
```

Funnel charts for pipeline/process stages:

```python
trace = go.Funnel(y=['Visit', 'Signup', 'Purchase', 'Retain'],
                  x=[1000, 300, 80, 50])
pyo.plot([trace])
```
