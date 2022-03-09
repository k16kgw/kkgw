import os
from typing import List

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt 
import matplotlib.pylab as p
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import animation, rc

from io import BytesIO
from PIL import Image

import pickle

class CahnHilliard_WithNeumannBC_ByDVDM():

    def __init__(
        self, 
        settings: dict = {
            'N': 10, # 内部領域の分割数
            'Dx': 0.5, # 領域の分割幅
            'Dt': 0.5, # 時間の分割幅
            'brank': 10, # 解を保存するステップ間隔
        },
        params: dict = {
            'Gamma': 2, # 拡散項の係数
            'const': 0.25, # 二重井戸型ポテンシャルの係数
        }
    ):
        self.settings = settings # space dimension
        self.params = params # parameters

    def Delta_f(self) -> np.ndarray:
        N = self.settings['N']

        Delta = np.zeros((N+2, N+2))
        Delta[0, 1:4] = [1, -2, 1]
        Delta[N+1, -4:-1] = [1, -2, 1]
        for k in range(1, N+1):
            Delta[k, k-1:k+2] = [1, -2, 1]
        return Delta

    def Mat_f(self) -> np.ndarray:
        """ 行列計算 """
        N = self.settings['N']

        Mat = np.zeros((N, N+2))
        Mat[0, 1:3] = [-2, 2]
        Mat[-1, -3:-1] = [2, -2]
        for k in range(1, N-1):
            Mat[k, k:k+3] = [1, -2, 1]
        return Mat

    def chem_func(self, U1, U2):
        """ 化学ポテンシャル """
        Dx = self.settings['Dx']
        Gamma = self.params['Gamma']
        const = self.params['const']

        Delta = self.Delta_f()
        output = Gamma/2/Dx**4 * np.dot(Delta, (U1 + U2)) \
            - const * (U2**3 + U2**2 * U1 + U2 * U1**2 + U1**3) \
            + 2*const * (U2 + U1)
        return output

    def equation(self, U2, U1) -> list:
        """ 方程式 """
        N = self.settings['N']
        Dx = self.settings['Dx']
        Dt = self.settings['Dt']

        Mat = self.Mat_f()

        eq = [0]*(N+4)
        # ノートでは k = -2, -1, 0, 1, ... , K-1, K, K+1 だが
        # ここでは k = 0, 1, 2, ... , K+1, K+2, K+3=Nx-1
        # 内部の U2 は k = 2, ... ,K+1

        eq[0] = U2[0] - U2[4]
        eq[1] = U2[1] - U2[3]
        eq[2:N+2] = U2[2:N+2] - U1[2:N+2] + Dt/Dx**2 * np.dot(Mat, self.chem_func(U1[1:N+3], U2[1:N+3]))
        eq[N+2] = U2[N+2] - U2[N]
        eq[N+3] = U2[N+3] - U2[N-1]

        return eq

    # def M_func(self, Utmp):
    def mass(self, Utmp) -> float:
        """ 質量 """
        N = self.settings['N']
        Dx = self.settings['Dx']

        output = (Utmp.sum() - Utmp[0]/2 - Utmp[N-1]/2) * Dx # cf. OFFY(2020)式(11)
        return output

    # def G_func(self, Utmp) -> List:
    def local_energy(self, Utmp) -> np.ndarray:
        """ 離散局所エネルギー """
        N = self.settings['N']
        Dx = self.settings['Dx']
        Gamma = self.params['Gamma']
        const = self.params['const']

        # G = [0]*N
        G = np.zeros(N)
        for k in range(2, N+2):
            i = k-2
            G[i] = const*(Utmp[k]**4 - 2*Utmp[k]**2 + 1) + Gamma/4/(Dx**2)*((Utmp[k+1]-Utmp[k])**2 + (Utmp[k]-Utmp[k-1])**2)
        return G

    def energy(self, G: np.ndarray) -> float:
        """
        離散全エネルギー
        Parameter
        ---------
        G: List
            離散局所エネルギー
        """
        N = self.settings['N']
        Dx = self.settings['Dx']

        # output = (sum(G) - G[0]/2 - G[N-1]/2) * Dx
        output = (G.sum() - G[0]/2 - G[N-1]/2) * Dx
        return output