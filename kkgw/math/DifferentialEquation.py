import numpy as np

class CahnHilliard_WithNeumannBC_ByDVDM():

    def __init__(
        self,
        settings: dict = {
            'N': 10, # 内部領域の分割数
            'Dx': 0.5, # 領域の分割幅
            'Dt': 0.5, # 時間の分割幅
        },
        params: dict = {
            'Gamma': 2, # 拡散項の係数
            'const': 0.25, # 二重井戸型ポテンシャルの係数
        }
    ):
        self.settings = settings # space dimension
        self.params = params # parameters

    def Laplacian(self, N) -> np.ndarray:
        output = np.zeros((N, N+2))
        for k in range(0, N):
            output[k, k:k+3] = [1, -2, 1]
        return output

    def chem_func(self, U1, U2) -> np.ndarray:
        """ 化学ポテンシャル """
        N = self.settings['N']
        Dx = self.settings['Dx']
        Gamma = self.params['Gamma']
        const = self.params['const']
        U1tmp = U1[1:N+3]
        U2tmp = U2[1:N+3]

        Lap = self.Laplacian(N+2)
        output = Gamma/2/Dx**2 * np.dot(Lap, (U1 + U2)) \
            - const * (U2tmp**3 + U2tmp**2 * U1tmp + U2tmp * U1tmp**2 + U1tmp**3) \
            + 2*const * (U2tmp + U1tmp)
        return output

    def equation(self, U2, U1) -> list:
        """ 方程式 """
        N = self.settings['N']
        Dx = self.settings['Dx']
        Dt = self.settings['Dt']

        Lap = self.Laplacian(N)

        eq = [0]*(N+4)
        # ノートでは k = -2, -1, 0, 1, ... , K-1, K, K+1 だが
        # ここでは k = 0, 1, 2, ... , K+1, K+2, K+3=Nx-1
        # 内部の U2 は k = 2, ... ,K+1

        eq[0] = U2[0] - U2[4]
        eq[1] = U2[1] - U2[3]
        eq[2:N+2] = U2[2:N+2] - U1[2:N+2] + Dt/Dx**2 * np.dot(Lap, self.chem_func(U1, U2))
        eq[N+2] = U2[N+2] - U2[N]
        eq[N+3] = U2[N+3] - U2[N-1]

        return eq

    def mass(self, Utmp) -> float:
        """ 質量 """
        N = self.settings['N']
        Dx = self.settings['Dx']

        output = (Utmp.sum() - Utmp[0]/2 - Utmp[N-1]/2) * Dx # cf. OFFY(2020)式(11)
        return output

    def local_energy(self, Utmp) -> np.ndarray:
        """ 離散局所エネルギー """
        N = self.settings['N']
        Dx = self.settings['Dx']
        Gamma = self.params['Gamma']
        const = self.params['const']

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
        G: np.ndarray
            離散局所エネルギー
        """
        N = self.settings['N']
        Dx = self.settings['Dx']

        # output = (sum(G) - G[0]/2 - G[N-1]/2) * Dx
        output = (G.sum() - G[0]/2 - G[N-1]/2) * Dx
        return output

class CahnHilliard_WithNeumannBC_ByFwdEuler():

    def __init__(
        self, 
        settings: dict = {
            'N': 10, # 内部領域の分割数
            'Dx': 0.5, # 領域の分割幅
            'Dt': 0.5, # 時間の分割幅
        },
        params: dict = {
            'Gamma': 2, # 拡散項の係数
            'const': 0.25, # 二重井戸型ポテンシャルの係数
        }
    ):
        self.settings = settings # space dimension
        self.params = params # parameters

    def Laplacian(self, N) -> np.ndarray:
        output = np.zeros((N, N+2))
        for k in range(0, N):
            output[k, k:k+3] = [1, -2, 1]
        return output

    def chem_func(self, U1):
        """ 化学ポテンシャル """
        N = self.settings['N']
        Dx = self.settings['Dx']
        Gamma = self.params['Gamma']
        const = self.params['const']

        Lap = self.Laplacian(N+2)
        output = Gamma/Dx**2 * np.dot(Lap, U1) \
            - const * 4*(U1[1:-1]**3 - U1[1:-1])
        return output

    def equation(self, U2, U1) -> list:
        """ 方程式 """
        N = self.settings['N']
        Dx = self.settings['Dx']
        Dt = self.settings['Dt']

        Lap = self.Laplacian(N)

        eq = [0]*(N+4)
        # ノートでは k = -2, -1, 0, 1, ... , K-1, K, K+1 だが
        # ここでは k = 0, 1, 2, ... , K+1, K+2, K+3=Nx-1
        # 内部の U2 は k = 2, ... ,K+1

        eq[0] = U2[0] - U2[4]
        eq[1] = U2[1] - U2[3]
        eq[2:N+2] = U2[2:N+2] - U1[2:N+2] + Dt/Dx**2 * np.dot(Lap, self.chem_func(U1))
        eq[N+2] = U2[N+2] - U2[N]
        eq[N+3] = U2[N+3] - U2[N-1]

        return eq

    def mass(self, Utmp) -> float:
        """ 質量 """
        N = self.settings['N']
        Dx = self.settings['Dx']

        output = (Utmp.sum() - Utmp[0]/2 - Utmp[N-1]/2) * Dx # cf. OFFY(2020)式(11)
        return output

    # def local_energy(self, Utmp) -> np.ndarray:
    #     """ 離散局所エネルギー """
    #     N = self.settings['N']
    #     Dx = self.settings['Dx']
    #     Gamma = self.params['Gamma']
    #     const = self.params['const']

    #     # G = [0]*N
    #     G = np.zeros(N)
    #     for k in range(2, N+2):
    #         i = k-2
    #         G[i] = const*(Utmp[k]**4 - 2*Utmp[k]**2 + 1) + Gamma/4/(Dx**2)*((Utmp[k+1]-Utmp[k])**2 + (Utmp[k]-Utmp[k-1])**2)
    #     return G

    # def energy(self, G: np.ndarray) -> float:
    #     """
    #     離散全エネルギー
    #     Parameter
    #     ---------
    #     G: List
    #         離散局所エネルギー
    #     """
    #     N = self.settings['N']
    #     Dx = self.settings['Dx']

    #     # output = (sum(G) - G[0]/2 - G[N-1]/2) * Dx
    #     output = (G.sum() - G[0]/2 - G[N-1]/2) * Dx
    #     return output