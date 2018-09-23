from bokeh.io import output_file, show
from bokeh.plotting import figure

# prepare some data
x = [3, 7.5, 10]
y = [3, 6, 9]

# prepare the output file
output_file("Line.html")

# create a figure object
f = figure()

# create line plot
f.triangle(x, y)

# write the plot in the figure object
show(f)

# Making a basic Bokeh line graph

# prepare some data
x = [3, 7.5, 10]
y = [3, 6, 9]

# prepare the output file
output_file("Line.html")

# create a figure object
f = figure()

# create line plot
f.circle(x, y)

# write the plot in the figure object
show(f)
