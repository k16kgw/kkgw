import json
import os
import pickle

import numpy as np
from scipy import optimize

class Calc1d():

    def __init__(
        self,
        cfg_inst, # cfgクラスのインスタンス
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
                timeset: dict = {
                    'inittime': 0, # 計算時の初期時刻
                    'timespan': 1000, # 計算時の時間ステップ数
                    'brank': 10, # 解を保存するステップ間隔
                    'plt_inittime': 0, # グラフプロットでの初期時刻
                    'plt_timespan': 1000, # グラフプロットでの時間ステップ数
                },
                output_dir = './data/output',
            ):
                self.settings = settings
                self.timeset = timeset
                self.output_dir = output_dir
            
            def initialdata(self, U):
                return 更新されたU
        """
        self.cfg = cfg_inst
        self.settings = self.cfg.settings
        self.timeset = self.cfg.timeset
        self.output_dir = self.cfg.output_dir
        self.initialdata = self.cfg.initialdata # function

        OUTPUT_VAR = os.path.join(self.output_dir, 'var')
        os.makedirs(OUTPUT_VAR, exist_ok=True)
        self.output_var = OUTPUT_VAR

        str_dt = str(self.settings['Dt'])
        if '.' in str_dt:
            self.dig = len(str_dt.split('.')[1]) # Dtの小数点以下の桁数
        elif 'e' in str_dt:
            self.dig = abs(int(str_dt.split('e')[1]))
        else:
            print('error')


    def preparation(self):
        """ 準備 """
        N = self.settings['N']

        OUTPUT_U = os.path.join(self.output_var, 'U')
        os.makedirs(OUTPUT_U, exist_ok=True)

        # 設定の保存
        with open(os.path.join(self.output_dir, 'cfg.pkl'), 'wb') as f:
            pickle.dump(self.cfg, f)
        with open(os.path.join(self.output_dir, 'cfg.json'), 'w') as f:
            json.dump(vars(self.cfg), f, indent=2)

        # 初期値の保存
        U = np.zeros((N, 2)) # 2ステップ分のU
        for idx in range(0, N):
            U[idx, 0] = self.initialdata(idx)
        np.save(os.path.join(OUTPUT_U, f't=0.0.npy'), U[:, 0])

    def calc(self, equation):

        N = self.settings['N']
        Dt = self.settings['Dt']
        inittime = self.timeset['inittime']
        timespan = self.timeset['timespan']
        brank = self.timeset['brank']

        # 初期値の読み出し
        U = np.zeros((N, 2))
        U[:, 0] = np.load(os.path.join(self.output_var, 'U', f't={inittime*Dt}.npy'))

        for t in range(inittime+1, inittime+timespan+1):
            U1 = U[:,0]
            # result = optimize.root(equation, U1, method="broyden1")
            result = optimize.root(equation, U1, args=U1, method="hybr")
            U[:,1] = result.x

            if t%brank==0 or t==(inittime+1):
                np.save(os.path.join(self.output_var, 'U', f't={round(t*Dt, self.dig)}.npy'), U[:,1])
                OUTPUT_dUdt = os.path.join(self.output_var, 'dUdt')
                os.makedirs(OUTPUT_dUdt, exist_ok=True)
                np.save(os.path.join(OUTPUT_dUdt, f't={round(t*Dt, self.dig)}.npy'), (U[:,1]-U[:,0])/Dt)
                if t%(brank*100)==0 or t==(inittime+1):
                    print(f't={round(t*Dt, self.dig)}')

            U[:, 0] = U[:, 1]