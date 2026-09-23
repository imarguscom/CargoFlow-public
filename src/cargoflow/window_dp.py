"""Bounded contiguous-block Held-Karp improvement, with full schedule checks."""
import ctypes
from pathlib import Path
import numpy as np

LIBRARY=Path(__file__).resolve().parents[2]/'artifacts/runtime_window_dp/window_dp.so'
_loaded=None


def improve_window_dp(route,travel,service,early,late,width,seconds,seed=42):
    global _loaded
    if _loaded is None:
        _loaded=ctypes.CDLL(str(LIBRARY))
        _loaded.cargoflow_window_dp.argtypes=[ctypes.c_int,np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')]+[
            np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')]*4+[
            ctypes.c_int,ctypes.c_double,ctypes.c_int,ctypes.POINTER(ctypes.c_int)]
        _loaded.cargoflow_window_dp.restype=ctypes.c_int
    travel=np.ascontiguousarray(travel,dtype=np.int64)
    n=len(travel)
    if travel.shape!=(n,n):raise ValueError('square matrix required')
    path=np.array(route,dtype=np.int32,copy=True)
    if len(path)!=n+1 or (path[0]!=0 or path[-1]!=0) or sorted(path[1:-1])!=list(range(1,n)):
        raise ValueError('complete depot tour required')
    arrays=[np.ascontiguousarray(x,dtype=np.int64) for x in [service,early,late]]
    if any(x.shape!=(n,) for x in arrays):raise ValueError('node array size mismatch')
    scans=ctypes.c_int()
    accepted=_loaded.cargoflow_window_dp(n,path,travel,*arrays,int(width),float(seconds),int(seed)%max(1,n),ctypes.byref(scans))
    return path.tolist(),dict(blocks_scanned=scans.value,accepted_blocks=accepted)
