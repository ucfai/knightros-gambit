"""Helper file for miscellaneous utility classes and functions.
"""
import platform

import numpy as np
from stockfish import Stockfish

def create_stockfish_wrapper():
    """Create simple wrapper around stockfish python module depending on operating system type.
    """
    operating_system = platform.system().lower()
    if operating_system == "darwin":
        stockfish_path = "/usr/local/bin/stockfish"
    elif operating_system == "raspi":
        stockfish_path = "n/a"
    elif operating_system == "linux":
        stockfish_path = "../chess-engine/stockfish_14.1_linux_x64/stockfish_14.1_linux_x64"
    elif operating_system == "windows":
        # stockfish_path = "../chess-engine/stockfish_14.1_win_x64_avx2.exe"
        stockfish_path = "C:/Users/juddb/Stockfish/stockfish_14.1_win_x64_avx2/stockfish_14.1_win_x64_avx2.exe"
    else:
        raise ValueError("Operating system must be one of "
                         "'darwin' (osx), 'linux', 'windows', 'raspi'")

    return Stockfish(stockfish_path)

def sig(value, scale):
    """Calculate the sigmoid of a value

    Attributes:
        value: The value to take the sigmoid of
        scale: How much to scale the value by
    """

    return 1 / (1 + np.exp(value/scale))
