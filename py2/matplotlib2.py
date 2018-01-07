import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
x = np.linspace(0,10,100)
y = np.cos(x)
z = np.sin(x)
# 2D Data or Images
data = 2 * np.random.random((10,10))
data2 = 3 * np.random.random((10, 10))
Y, X = np.mgrid[-3:3:100j,-3:3:100]
U = -1 - X**2 + Y
V = 1 + X - Y**2
from matplotlib.cbook import get_sample_data
#img = np.load(get_sample_data('axes_grid/bivariate_normal.npy'))
# 2 Create Plot
#import matplotib.pyplot as plt
fig = plt.figure()
fig2 = plt.figure(figsize=plt.figaspect(2.0))
# Axes
"""
All plotting is done with respect to an Axes.In most
cases,a subplot will fit your needs.A subplot is an
axes on a grid system.
"""
fig.add_axes()
ax1 = fig.add_subplot(221)    # row-col-num
ax3 = fig.add_subplot(212)
fig3, axes = plt.subplots(nrows=2, ncols=2)
fig4, axes2 = plt.subplots(ncols=3)

# 3 Plotting Routines
# 1D Data
fig, ax = plt.subplots()
lines = ax.plot(x,y)                 # Draw points with lines or markers connecting them
ax.scatter(x,y)                      # Draw unconnected points,scaled or colored
axes[0,0].bar([1,2,3],[3,4,5])       # Plot vertical rectangles (constant width)
axes[1,0].barh([0.5,1,2.5],[0,1,2])  # Plot horizontal rectangles (cnostant width)
axes[1,1].axhline(0.45)              # Draw a horizontal line across axes
axes[0,1].axvline(0.65)              # Draw a vertical line across axes
ax.fill(x,y,color='blue')            # Draw filled polygons
ax.fill_between(x,y,color='yellow')  #Fill between y-values and o
# 2D Data or Images
fig, ax = plt.subplots()
"""
im = ax.imshow(img,                    # Colormapped or RGB arrays
               cmap='gist_earth',
               interpolation='nearest',
               vmin=-2,
               vmax=2)
"""
# 4 Customize Plot
plt.plot(x, x, x,x**2, x, x**3)
ax.plot(x,y,alpha = 0.4)
ax.plot(x, y, c='k')
#fig.colorbar(im, orientation='horizontal')
"""
im = ax.imshow(img,
               cmap='seismic')
"""
# Markers
fig, ax = plt.subplots()
ax.scatter(x, y, marker='.')
ax.plot(x, y,marker="o")

# Linestyles
plt.plot(x, y, linewidth=4.0)
plt.plot(x, y, ls='solid')
plt.plot(x, y,ls='--')
plt.plot(x, y, '--', x**2,y**2,'-.')
plt.setp(lines,color='r', linewidth=4.0)

# Text & Annotations
ax.text(1,
        -2.1,
        'Example Graph',
        style='italic')
ax.annotate("Sine",
            xy=(8, 0),
            xycoords='data',
            xytext=(10.5, 0),
            textcoords='data',
            arrowprops=dict(arrowstyle="->",
                        connectionstyle="arc3"),)
# Vector Fields
axes[0,1].arrow(0,0,0.5,0.5)      # Add an arrow to the axes
#axes[1,1].quiver(y,z)             # Plot a 2D field of arrows
#axes[0,1].streamplot(X,Y,U,V)      # Plot a 2D field or arrows
# Data Distributions
ax1.hist(y)                       # Plot a histogram
ax3.boxplot(y)                    # Make a box and whisker plot
ax3.violinplot(z)                 # Make a violin plot
# Data Distributions
ax1.hist(y)                       # Plot a histogram
ax3.boxplot(y)                    # Make a box and whisker plot
ax3.violinplot(x)                 # Make a violin plot
axes2[0].pcolor(data2)            # Pseudocolor plot of 2D array
axes2[0].pcolormesh(data)         # Pseudocolor plot of 2D array
CS = plt.contour(Y, X,U)          # Plot contours
axes2[2].contourf(data1)        # Plot filled contours
axes2[2] = ax.clabel(CS)        #  Label a contour plot

# Workflow
#The basic steps to creating plots with matplotlib are:
# 1 Prepare data 2 Create plot 3 Plot 4 Customize plot 5 Save plot 6 Show plot
x = [1,2,3,4]
y = [10,20,25,30]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,y, color='lightblue',linewidth=3)
ax.scatter([2,4,6],
           [5,15,25],
           color='darkgreen',
           marker="^")
ax.set_xlim(1,6.5)
plt.savefig('foo.png')
plt.show()
