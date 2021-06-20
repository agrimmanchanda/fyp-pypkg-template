"""
Plotting greetings
============================

Example using your package
"""

# Libraries
from labimputer.utils.display import show
from labimputer.core.greetings import Hello
from labimputer.core.greetings import Morning

# -------------------------------------
# Constants
# -------------------------------------

# -------------------------------------
# Main
# -------------------------------------
# Execute show
show()

# Create instances
h = Hello()
m = Morning()

# Greet
h.greet(name='Maria')
m.greet(name='Damien')