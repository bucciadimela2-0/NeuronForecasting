import numpy as np


class Constants:
    N = 20  
    A1 = np.array([
        [5.25677, 3.22776, 0.02343, 1.00899, 0.86886],
        [3.22776, 4.77906, 0.71110, 1.58785, 0.68990],
        [0.02343, 0.71110, 5.39732, 1.27769, 1.03968],
        [1.00899, 1.58785, 1.27769, 3.83577, 1.92157],
        [0.86886, 0.68990, 1.03968, 1.92157, 4.69323]
    ])

    FN = 'FitzhughNagumoModel'
    IZH = 'IzhikevicModel'
    LIF = 'LifModel'
    v = "v - Membrane potential"
    u = "u - Refractary State"

