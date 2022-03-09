import json
import os

import numpy as np
from scipy import optimize

class Calc():

    def __init__(
        self,
        settings: dict = {
            'N': 10, # 内部領域の分割数
            'Dx': 0.5, # 領域の分割幅
            'Dt': 0.5, # 時間の分割幅
        },
        initialdata: dict = {
            'a0': 0.01, # 初期値の係数
            'wn': 4, # 波数
            'func': 'a0 * np.cos(wn * np.pi*(idx)/(N+5))', # 関数形
        },
        timeset: dict = {
            'inittime': 0,
            'timespan': 1000,
            'brank': 100, # 解を保存するステップ間隔
            'Dt': 0.5 # settingsと同じ値
        },
        output_dir: str='./data/output'
    ):
        self.settings = settings # space dimension
        self.initialdata = initialdata
        self.timeset = timeset
        self.output_dir = output_dir

        OUTPUT_VAR = self.output_dir / 'var'
        OUTPUT_VAR.mkdir(parents=True, exist_ok=True)
        self.output_var = OUTPUT_VAR

    def preparation(self):
        """ 準備 """
        N = self.settings['N']
        a0 = self.initialdata['a0']
        wn = self.initialdata['wn']

        OUTPUT_U = self.output_var / 'U'
        OUTPUT_U.mkdir(parents=True, exist_ok=True)

        # パラメタの保存
        with open(os.path.join(self.output_dir, 'settings.json'), 'w') as fp:
            json.dump(self.settings, fp)
        with open(os.path.join(self.output_dir, 'initialdata.json'), 'w') as fp:
            json.dump(self.initialdata, fp)

        # 初期値の保存
        U = np.zeros((N+4, 2)) # 2ステップ分のU
        for idx in range(0, N+4):
            # 初期条件
            U[idx, 0] = eval(self.initialdata['func'])
        # 初期値の保存
        np.save(os.path.join(OUTPUT_U, f't=0.0.npy'), U[:, 0])

    def calc(self, equation):
        N = self.settings['N']
        inittime = self.timeset['inittime']
        timespan = self.timeset['timespan']
        brank = self.timeset['brank']
        Dt = self.timeset['Dt']

        # 初期値の読み出し
        U = np.zeros((N+4, 2))
        U[:, 0] = np.load(os.path.join(self.output_var, 'U', f't={inittime*Dt}.npy'))

        for t in range(inittime+1, inittime+timespan+1):
            U1 = U[:,0]
            # result = optimize.root(equation, U1, method="broyden1")
            result = optimize.root(equation, U1, args=U1, method="hybr")
            U[:,1] = result.x

            if t%brank==0 or t==(inittime+1):
                np.save(os.path.join(self.output_var, 'U', f't={t*Dt}.npy'), U[:,1])
                OUTPUT_dUdt = self.output_var / 'dUdt'
                OUTPUT_dUdt.mkdir(parents=True, exist_ok=True)
                np.save(os.path.join(OUTPUT_dUdt, f't={t*Dt}.npy'), (U[:,1]-U[:,0])/Dt)
                if t%(brank*100)==0 or t==(inittime+1):
                    print(f't={t*Dt}')

            U[:, 0] = U[:, 1]