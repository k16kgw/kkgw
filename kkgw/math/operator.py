from mimetypes import init
import numpy as np

def l(N: init):
    """ ラプラシアン """
    Delta = np.zeros((N+2, N+2))
    Delta[0, 1:4] = [1, -2, 1]
    Delta[N+1, -4:-1] = [1, -2, 1]
    for n in range(1, N+1):
        Delta[n, n-1:n+2] = [1, -2, 1]
    return Delta
