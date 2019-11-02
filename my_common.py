import os
import sys

import numpy as np

def noneFromShape(shape):
    if (len(shape) == 0):
        return None

    shape = list(shape)
    ll = shape.pop(0)

    ret = []
    for i in range(ll):
        ret.append(noneFromShape(shape))

    return ret


def isListOrArray(aa):
    if isinstance(aa, np.ndarray):
        if len(aa.shape) == 0:
            return False
        else:
            return True
    return isinstance(aa, list) or isinstance(aa, tuple) or isinstance(aa, np.matrix)


def isDict(aa):
    if isinstance(aa, dict):
        return True
    else:
        return False


def print_exception():
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print("\nEXCEPTION!")
    print(exc_type, exc_obj)
    print(fname, exc_tb.tb_lineno)
