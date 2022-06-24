import numpy as np

class CahnHilliardEq_NeumannBC_1d_DVDM():

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

class CahnHilliardEq_NeumannBC_1d_FwdEuler():

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
        G: np.ndarray
            離散局所エネルギー
        """
        N = self.settings['N']
        Dx = self.settings['Dx']

        output = (G.sum() - G[0]/2 - G[N-1]/2) * Dx
        return output

class HeatEq_NeumannBC_1d_DVDM():
    """
    内部：熱方程式
    境界：斉次ノイマン
    領域：1次元
    """

    def __init__(self, cfg_inst):
        """
        CFG クラスの構成
        ======
        class CFG:
            def __init__(
                self,
                date = '220623',
                ver = 'fw-6',
                settings: dict = {
                    'N': 100, # 内部領域の分割数
                    'Dx': 0.001, # 領域の分割幅
                    'Dt': 0.0001, # 時間の分割幅
                },
                params: dict = {
                    'Gamma': 2,
                },
                timeset: dict = {
                    'inittime': 0, # 計算時の初期時刻
                    'timespan': 5000, # 計算時の時間ステップ数
                    'brank': 10, # 解を保存するステップ間隔
                    'plt_inittime': 0, # グラフプロットでの初期時刻
                    'plt_timespan': 5000, # グラフプロットでの時間ステップ数
                },
                output_dir = './data/output',
            ):
                self.settings = settings
                self.params = params
                self.timeset = timeset
                self.output_dir = os.path.join(output_dir, date+'-'+ver)
            
            def initialdata(self, x):
                return 関数(x)
        """
        self.cfg = cfg_inst
        self.settings = self.cfg.settings # space dimension
        self.params = self.cfg.params # parameters

    def Laplacian(self, N) -> np.ndarray:
        """ ラプラシアン """
        output = np.zeros((N, N))
        for k in range(1, N-1):
            output[k, k-1:k+2] = [1, -2, 1]
        return output

    def equation(self, U2, U1) -> list:
        """
        DVDM
        境界ぴったり
        """
        N = self.settings['N']
        Dx = self.settings['Dx']
        Dt = self.settings['Dt']
        Gamma = self.params['Gamma']

        Lap = self.Laplacian(N)

        eq = [0]*(N)
        eq[0] = U2[1]-U2[0] # eq[0] = - Gamma*(U1[1]-U1[0])/Dx
        eq[1:N-1] = (U2[1:N-1]-U1[1:N-1])*Dx**2 - Gamma*0.5*np.dot(Lap, (U1+U2))[1:N-1]*Dt # (U2[1:N-1]-U1[1:N-1])/Dt - Gamma*np.dot(Lap, (U1+U2))[1:N-1]/(Dx**2)/2
        eq[N-1] = U2[N-1]-U2[N-2] # eq[N-1] = + Gamma*(U1[N-1]-U1[N-2])/Dx

        return eq


class HeatEq_ReactionBC_1d_DVDM():
    """
    内部：熱方程式
    境界：反応方程式
    領域：1次元
    """

    def __init__(self, cfg_inst):
        """
        CFG クラスの構成
        ======
        class CFG:
            def __init__(
                self,
                date = '220623',
                ver = 'fw-6',
                settings: dict = {
                    'N': 100, # 内部領域の分割数
                    'Dx': 0.001, # 領域の分割幅
                    'Dt': 0.0001, # 時間の分割幅
                },
                params: dict = {
                    'Gamma': 2,
                },
                timeset: dict = {
                    'inittime': 0, # 計算時の初期時刻
                    'timespan': 5000, # 計算時の時間ステップ数
                    'brank': 10, # 解を保存するステップ間隔
                    'plt_inittime': 0, # グラフプロットでの初期時刻
                    'plt_timespan': 5000, # グラフプロットでの時間ステップ数
                },
                output_dir = './data/output',
            ):
                self.settings = settings
                self.params = params
                self.timeset = timeset
                self.output_dir = os.path.join(output_dir, date+'-'+ver)
            
            def initialdata(self, x):
                return 関数(x)
        """
        self.cfg = cfg_inst
        self.settings = self.cfg.settings # space dimension
        self.params = self.cfg.params # parameters

    def Laplacian(self, N) -> np.ndarray:
        """ ラプラシアン """
        output = np.zeros((N, N))
        for k in range(1, N-1):
            output[k, k-1:k+2] = [1, -2, 1]
        return output
    
    def Delx(self, Ubd, Uin):
        """ 法線微分(向きも考慮されている) """
        output = Uin - Ubd
        return output

    def Reaction(self, U2, U1):
        """ 反応項 """
        N = self.settings['N']
        Dx = self.settings['Dx']
        const = self.params['const']
        
        output = const * (
            (U2**3 + U2**2*U1 + U2*U1**2 + U1**3)
            - (U2 + U1)
        )
        return output

    def equation(self, U2, U1) -> list:
        """ 方程式 """
        N = self.settings['N']
        Dx = self.settings['Dx']
        Dt = self.settings['Dt']
        Gamma = self.params['Gamma']

        Lap = self.Laplacian(N)

        eq = [0]*(N)

        eq[0] = (U2[0]-U1[0])*Dx - self.Reaction(U2[0], U1[0])*Dt*Dx + Gamma*0.5*(self.Delx(Ubd=U2[0]+U1[0], Uin=U2[1]+U1[1]))*Dt
        eq[1:N-1] = (U2[1:N-1]-U1[1:N-1])*Dx**2 - Gamma*0.5*np.dot(Lap, (U1+U2))[1:N-1]*Dt # (U2[1:N-1]-U1[1:N-1])/Dt - Gamma*np.dot(Lap, (U1+U2))[1:N-1]/(Dx**2)/2
        eq[N-1] = (U2[N-1]-U1[N-1])*Dx - self.Reaction(U2[N-1], U1[N-1])*Dt*Dx + Gamma*0.5*(self.Delx(Ubd=U2[N-1]+U1[N-1], Uin=U2[N-2]+U1[N-2]))*Dt

        return eq
