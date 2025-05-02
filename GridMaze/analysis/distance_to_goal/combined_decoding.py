"""
New Lib for combining decoding analyses to see if goal decoding at cue improve when using decoders that know
about distance to goal while controlling for place coding in the neuronal population.
@peterdoohan
"""

# %% Imports

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from GridMaze.analysis.core import get_sessions as gs
from . import place_decoding as dp
from . import decoding_utils as du
from . import goal_decoding as gd
from . import bases


# %% Global Variables


# %% Functions


def test():
    """ """

    return
