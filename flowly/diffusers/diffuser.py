from dataclasses import dataclass, field

import numpy as np


@dataclass
class Diffuser:
    shroud_line: np.ndarray = field(init=False)
    "line that represents the shroud of the diffuser"

    bottom_line: np.ndarray = field(init=False)
    "line that represents the shroud of the diffuser"


