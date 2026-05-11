import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
# from pyodide.http import open_url
import os

url = "https://raw.githubusercontent.com/JosephBARBIERDARNAL/data-matplotlib-journey/refs/heads/main/natural-disasters/natural-disasters.csv"
df = pd.read_csv(url)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.style'] = 'italic'
fig, ax = plt.subplots(figsize=(10, 6), layout="tight")

ax.axis(False)
n_lines = len(df.columns) # this will be the number of vertical lines for the parallel plot
margin = 3
for i in range(1, n_lines):
    ax.axvline(i*margin/n_lines, color='#949494', linestyle='--', linewidth=1, zorder=-1)

cmap = mpl.colormaps['magma']
years = df['Year'].tolist()
n_years = len(years)
colors = cmap(np.linspace(0, 1, n_years))

vlines_names = df.columns.to_list()

# add title
fig.text(.5, 1, "Natural disasters have become more frequent recently!", horizontalalignment='center')

# add plotting and legents for the year
min_year, max_year = min(years), max(years)
ax.scatter([0]*2*n_years, np.linspace(0, 1, 2*n_years), marker='s', s=100, c=cmap(np.linspace(0, 1, 2*n_years)), linewidth=3)
ax.text(-.05, 1.08, 'Year', fontsize=8, verticalalignment='bottom')
ax.text(-0.04, 0, str(min_year), fontsize=7, horizontalalignment='right')
ax.text(-0.04, 1, str(max_year), fontsize=7, horizontalalignment='right')

for i in range(0, n_lines):

    # get values
    name = vlines_names[i]
    values = df[name].values.tolist()
    minv = min(values)
    maxv = max(values)
    norm_values = [(v-minv)/(maxv-minv) for v in values]
    yticks = [str(v) for v in values]
    

    if name != 'Year': # already added yticks for 'Year'
        n_words = len(name.split())
        mx = max([len(nm) for nm in name.split()])
        offset = mx/2*0.015 # slightly center the name under the vertical line
        name = '\n'.join(name.split()) # TODO: optimize text wrapping
        xpos =i*margin/n_lines
        # add feature label on top plus min/max values
        ax.text(xpos-offset, 1.08, name, fontsize=8, verticalalignment='bottom') # also slightly center y axis in case of multi-row name
        ax.text(xpos-.007, -.03, str(int(minv)), fontsize=7, horizontalalignment='right', color='black')
        ax.text(xpos-.007, 1.03, str(int(maxv)), fontsize=7, horizontalalignment='right', color='black')
    
        # add data
        ax.scatter([xpos]*len(norm_values), norm_values, c=colors)

    # add lines connecting previous feature points with current one
    if i == 0:
        continue
        
    prev_name = vlines_names[i-1]
    prev_values = df[prev_name].values.tolist()
    prev_norm_values = [(v-min(prev_values))/(max(prev_values)-min(prev_values)) for v in prev_values]
    xpos_prev = (i-1)*margin/n_lines
    for k in range(len(values)):
        x = [xpos_prev, xpos]
        y = [prev_norm_values[k], norm_values[k]]
        ax.plot(x, y, color=colors[k], linewidth=.3)

# Get current directory and save the plot
current_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(current_dir, "parallel_plot-natural_disasters.svg"), bbox_inches='tight', dpi=300)
plt.show()