import os
from django.conf import settings

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt 
import matplotlib.pylab as p
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import animation, rc

from io import BytesIO
from PIL import Image

import pickle

from kkgw.math.DifferentialEquation import CahnHilliard

class Calc():

    def __init__(
        self,
        settings: dict = {
            'N': 10, # 内部領域の分割数
            'Dx': 0.5, # 領域の分割幅
            'Dt': 0.5, # 時間の分割幅
            'brank': 100, # 解を保存するステップ間隔
        },
        initialdata: dict = {
            'a0': 0.01, # 初期値の係数
            'wn': 4, # 波数
            'func': 'a0 * np.cos(wn * np.pi*(idx)/(N+5))', # 関数形
        },
        output_dir: str='./data/output'
    ):
        self.settings = settings # space dimension
        self.initialdata = initialdata
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
        with open(os.path.join(self.output_dir, 'settings.json'), 'wb') as fp:
            pickle.dump(self.settings, fp)
        with open(os.path.join(self.output_dir, 'initialdata.json'), 'wb') as fp:
            pickle.dump(self.initialdata, fp)

        # 初期値の保存
        U = np.zeros((N+4, 2)) # 2ステップ分のU
        for idx in range(0, N+4):
            # 初期条件
            U[idx, 0] = eval(self.initialdata['func'])
        # 初期値の保存
        t = 0
        np.save(os.path.join(OUTPUT_U, f'U_t={t}.npy'), U[:, 0])

    def calc(self, equation, init_time=0, timespan=100):
        N = self.settings['N']
        brank = self.settings['brank']

        # 初期値の読み出し
        U = np.zeros((N+4, 2))
        U[:, 0] = np.load(os.path.join(self.output_var, 'U', f'U_t={init_time}.npy'))

        for t in range(init_time+1, init_time+timespan+1):
            U1 = U[:, 0]
            # result = optimize.root(equation, U1, method="broyden1")
            result = optimize.root(equation, U1, args=U1, method="hybr")
            # result = optimize.root(CahnHilliard.equation, U1, args=U1, method="hybr")
            U[:, 1] = result.x

            if t%brank==0 or t==(init_time+1):
                np.save(os.path.join(self.output_var, 'U', f'U_t={t}.npy'), U[:, 1])
                if t%(brank*100)==0 or t==(init_time+1):
                    print(f't={t}')

            U[:, 0] = U[:, 1]