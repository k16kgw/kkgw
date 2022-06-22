import json
import os
import pickle

import numpy as np
from scipy import optimize

class Calc1d():

    def __init__(
        self,
        CFG,
    ):
        """
        CFG クラスの構成
        ======
        class CFG:
            def __init__(
                self,
                settings = {
                    'N': 100, # 内部領域の分割数
                    'Dx': 0.5, # 領域の分割幅
                    'Dt': 0.5, # 時間の分割幅
                },
                output_dir = './data/output',
            ):
                self.settings = settings
                self.output_dir = output_dir
            
            def initialdata(self, x):
                return x
        """
        self.CFG = CFG()
        self.settings = CFG().settings
        self.output_dir = CFG().output_dir
        self.initialdata = CFG().initialdata # function

        OUTPUT_VAR = os.path.join(self.output_dir, 'var')
        os.makedirs(OUTPUT_VAR, exist_ok=True)
        self.output_var = OUTPUT_VAR

    def preparation(self):
        """ 準備 """
        N = self.settings['N']

        OUTPUT_U = os.path.join(self.output_dir, 'U')
        os.makedirs(OUTPUT_U, exist_ok=True)

        # 設定の保存
        with open(os.path.join(self.output_dir, 'CFG.pkl'), 'wb') as f:
            pickle.dump(self.CFG, f)
        with open(os.path.join(self.output_dir, 'CFG.json'), 'w') as f:
            json.dump(vars(self.CFG), f, indent=2)

        # 初期値の保存
        U = np.zeros((N, 2)) # 2ステップ分のU
        for x in range(0, N):
            U[x, 0] = self.initialdata(x)
        np.save(os.path.join(OUTPUT_U, f't=0.0.npy'), U[:, 0])

    def calc(
        self,
        equation,
        timeset: dict = {
            'inittime': 0,
            'timespan': 1000,
            'brank': 100, # 解を保存するステップ間隔
            'Dt': 0.5 # settingsと同じ値
        },
    ):
        N = self.settings['N']
        inittime = timeset['inittime']
        timespan = timeset['timespan']
        brank = timeset['brank']
        Dt = timeset['Dt']

        # 初期値の読み出し
        U = np.zeros((N, 2))
        U[:, 0] = np.load(os.path.join(self.output_var, 'U', f't={inittime*Dt}.npy'))

        for t in range(inittime+1, inittime+timespan+1):
            U1 = U[:,0]
            # result = optimize.root(equation, U1, method="broyden1")
            result = optimize.root(equation, U1, args=U1, method="hybr")
            U[:,1] = result.x

            if t%brank==0 or t==(inittime+1):
                np.save(os.path.join(self.output_var, 'U', f't={t*Dt}.npy'), U[:,1])
                OUTPUT_dUdt = os.path.join(self.output_var, 'dUdt')
                os.makedirs(OUTPUT_dUdt, exist_ok=True)
                np.save(os.path.join(OUTPUT_dUdt, f't={t*Dt}.npy'), (U[:,1]-U[:,0])/Dt)
                if t%(brank*100)==0 or t==(inittime+1):
                    print(f't={t*Dt}')

            U[:, 0] = U[:, 1]