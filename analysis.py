# %%
"""

# Iris Data Set

#Analysis proram script

# Data soure: [archive.ics.uci.edu](https://archive.ics.uci.edu/dataset/53/iris)

"""
# Libraries

# info obtained from the class Principles of Data Analytics t08v02_load_iris on March 11, 2024 

# Data Frames.

import pandas as pd

# Plotting.
import matplotlib.pyplot as plt

# Numerical arrays.
import numpy as np

# Import seaborn.

import seaborn as sns

# Load Data

df = pd.read_csv('https://raw.githubusercontent.com/h2ohugol/Pands-myproject/main/iris.data', 
                 names=['Sepal Length','Sepal Width','Petal Length','Petal Width', 'Species'])

# Have a look.

df

# %%

# Analyse Data.

df.dtypes

# Review

df.describe()



# Two variables: Petal Length and Petal width.


# Petal length.
plen = df['Petal Length']

# Show.
print(plen)

# Type.
print(type(plen))

# The numpy array.
plen = plen.to_numpy()

# Show.
plen

# Petal width.
pwidth = df['Petal Width'].to_numpy()

# Show.
pwidth

# Plots

# Basic plot.
plt.plot(plen, pwidth, '*', color='skyblue', markeredgecolor='blue')

# Axis labels.
plt.xlabel('Petal Length (cm)', fontsize=12, color='black')
plt.ylabel('Petal Width (cm)', fontsize=12, color='black')

# Title.
plt.title('Iris Data Set', fontsize=15, color='blue')

# X limits.
plt.xlim(0, 8)

# Y limits.
plt.ylim(0, 4)

# Adding a best fit line


# y = mx + c = p_1 x^1 + p_0 = p_1 x + p_0

# This is a equation of a straight line in slope-intercept form, y = mx + c where:

#  y is the dependant variable (vertical axis in the graph).
#  x is the independent variable (horizontal axis in the graph).
#  m is the slope of the line, which represents the rate of change of y with respect to x.
#  c is the y-intercept of the line, which is the value of y when x = 0.
#  In the second part of the equation, the variables 𝑝1 and 𝑝0 are coefficients representing
#  the slope and y-intercept, respectively. They correspond to 𝑚 and 𝑐 in the first part of the equation.

# Fit a straight line between x and y.
m, c = np.polyfit(plen, pwidth, 1)

# Show m and c.
m, c

# Design of a new plot and set of axes.
fig, ax = plt.subplots()

# Basic plot.
ax.plot(plen, pwidth, '*', color='skyblue', markeredgecolor='blue')

# Basic plot.
ax.plot(plen, m * plen + c, 'r-')

# Axis labels.
ax.set_xlabel('Petal Length (cm)', fontsize=12, color='black')
ax.set_ylabel('Petal Width (cm)', fontsize=12, color='black')

# Title.
ax.set_title('Iris Data Set', fontsize=15, color='blue')

# X limits.
ax.set_xlim(0, 8)

# Y limits.
ax.set_ylim(0, 4)

# x values for best fit line.
bf_x = np.linspace(0.0, 8.0, 100)

# y values for best fit line.
bf_y = m * bf_x + c

# Design of a new plot and set of axis.
fig, ax = plt.subplots()

# Basic plot.
ax.plot(plen, pwidth, '*', color='skyblue', markeredgecolor='blue')

# Basic plot.
ax.plot(bf_x, bf_y, 'r-')

# Axis labels.
ax.set_xlabel('Petal Length (cm)', fontsize=12, color='black')
ax.set_ylabel('Petal Width (cm)', fontsize=12, color='black')

# Title.
ax.set_title('Iris Data Set', fontsize=15, color='blue')

# X limits.
ax.set_xlim(0, 8)

# Y limits.
ax.set_ylim(-1, 4)

# The messure of the correlation coefficient.
np.corrcoef(plen, pwidth)

# Seaborn bar chart visualization.

# Group by species and calculate mean for each feature.
df_grouped = df.groupby('Species').mean()

# Plotting.
plt.figure(figsize=(10, 6))
sns.set_palette('pastel')
df_grouped.plot(kind='bar' , figsize=(10, 6))

# Title and labels.
plt.title('Mean Feature Values by Species', fontsize=16)
plt.xlabel('Species', fontsize=14)
plt.ylabel('Mean Value', fontsize=14)

# Show plot.
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Seaborn histogram visualization.

# Set the style of the seaborn plots.
sns.set(style="whitegrid")

# Plot histograms for each feature.
plt.figure(figsize=(12, 8))
for i, feature in enumerate(df.columns[:-1]):
    plt.subplot(2, 2, i+1)
    sns.histplot(data=df, x=feature, kde=True, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {feature}', fontsize=14)
    plt.xlabel('')
    plt.ylabel('Frequency', fontsize=12)


plt.tight_layout()
plt.show()


# %%
"""
## References and sources

* Add names to the Data Set [STATOLOGY](https://www.statology.org/pandas-dataframe-header/#:~:text=You%20can%20use%20one%20of%20the%20following%20three,df%20%3D%20pd.DataFrame%28data%3D%5Bdata_values%5D%29%20df.columns%20%3D%20%5B%27A%27%2C%20%27B%27%2C%20%27C%27%5D)

* An introduction to [seaborn](https://seaborn.pydata.org/tutorial/introduction)

* Seaborn [Iris data set](https://github.com/mwaskom/seaborn-data/blob/master/iris.csv)

* Understanding matplotlib [Real python](https://realpython.com/python-matplotlib-guide/#understanding-pltsubplots-notation)

* Pearson corelation [Laerd statistics](https://statistics.laerd.com/statistical-guides/pearson-correlation-coefficient-statistical-guide.php)

* Pearson Correlation coefficient [Wikipedia](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)

* Correlation coefficient [numpy](https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html)

* Fit a polynomial [numpy.polyfit](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html)

"""
