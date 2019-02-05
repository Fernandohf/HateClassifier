"""
Script that creates the index.html widget to test the model.
"""
from model import *
import numpy as np
from bokeh import events
from bokeh.io import output_file
from bokeh.models import Div, ColumnDataSource
from bokeh.models.widgets import TextInput
from bokeh.plotting import figure
from bokeh.layouts import column, row, CustomJS

# Prepares the data
default_text = "I wish you a lovely day!"


# Create a new plot without tools
p = figure(plot_width=800, plot_height=600, tools=None)
p.vbar()
# Input text widget
text_input = TextInput(value="Have a lovely day!", title="Comment:")

# Custom JS to update graph
update_graph = CustomJS(args=dict(source=source, slider=slider), code="""
    var data = source.data;
    var f = slider.value;
    x = data['x']
    y = data['y']
    for (i = 0; i < x.length; i++) {
        y[i] = Math.pow(x[i], f)
    }

    // necessary becasue we mutated source.data in-place
    source.change.emit();
""")

text_input.