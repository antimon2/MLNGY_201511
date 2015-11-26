#!/usr/bin/env python
# -*- coding: utf-8 -*-
# chk_python.py

import sys
print("Python: v%d.%d.%d" % (sys.version_info.major, sys.version_info.minor, sys.version_info.micro))


try:
    import numpy as np
    # print("NumPy: v%s" % np.version.version__)
    print("NumPy: v%s" % np.__version__)
except ImportError:
    print("NumPy: -")


try:
    import matplotlib as mp
    print("matplotlib: v%s" % mp.__version__)
except ImportError:
    print("matplotlib: -")


try:
    import IPython as ip
    print("IPython: v%s" % ip.__version__)
except ImportError:
    print("IPython: -")
