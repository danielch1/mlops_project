import os
import sys

_DATA_ROOT = os.path.dirname(__file__)  # root of data
_SRC_ROOT = os.path.dirname(_DATA_ROOT)  # root of src
_PROJECT_ROOT = os.path.dirname(_SRC_ROOT)  # root of project
sys.path.append(_PROJECT_ROOT)
