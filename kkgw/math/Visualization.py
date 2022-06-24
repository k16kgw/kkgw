import os
import json

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.pylab as p
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import animation, rc

from io import BytesIO
from PIL import Image

def close_fig(fig, close):
    if close:
        plt.close(fig)
    else:
        plt.show()

class Plot2d():
    def __init__(self, output_dir:str='./data/output'):
        self.output_dir = output_dir
        OUTPUT_FIG = os.path.join(output_dir, 'fig')
        os.makedirs(OUTPUT_FIG, exist_ok=True)
        self.output_fig = OUTPUT_FIG

    def snapshot(self, fpl, name:str, time, close=True):
        """ 各時刻での空間x関数のグラフ """
        OUTPUT_FIG_plot = os.path.join(self.output_fig, name)
        os.makedirs(OUTPUT_FIG_plot, exist_ok=True)

        fig = plt.figure(figsize=(6,5), facecolor='w')
        ax = fig.add_subplot(
            title=f'time={time}',
            xlabel='Position',
            ylabel=name
        )
        x = list(range(0, fpl.size))
        ax.plot(x, fpl)
        fig.savefig(os.path.join(OUTPUT_FIG_plot, f't={time}.png'))
        close_fig(fig, close)

    def timeseries(self, varname: str, timeset: dict, close=True):
        fpl = np.load(os.path.join(self.output_dir, 'var', f'{varname}.npy'))

        fig = plt.figure(figsize=(6,5), facecolor='w')
        ax = fig.add_subplot(111)
        t = np.linspace(timeset['inittime'], int(timeset['timespan']*timeset['Dt']), int(timeset['timespan']/timeset['brank'])+1)
        plt.plot(t, fpl, color='r')
        ax.set_xlabel('time')
        ax.set_ylabel(varname)
        ax.set_title(varname)
        fig.savefig(os.path.join(self.output_fig, f'{varname}.png'))
        close_fig(fig, close)

class Anim2d():
    """
    フォルダ階層
    {output_dir}---var---{varname}.npy
                 |     |-{varname}---t=0.npy <-
                 |-fig
                 |-settings.json
    """
    def __init__(self, cfg_inst):
        """
        CFG クラスの構成
        ======
        class CFG:
            def __init__(
                self,
                settings: dict = {
                    'N': 100, # 内部領域の分割数
                    'Dx': 0.5, # 領域の分割幅
                    'Dt': 0.5, # 時間の分割幅
                },
                timeset: dict = {
                    'inittime': 0,
                    'timespan': 1000,
                    'brank': 10, # 解を保存するステップ間隔
                    'plt_inittime': 0, # グラフプロットでの初期時刻
                    'plt_timespan': 1000, # グラフプロットでの時間ステップ数
                },
                output_dir = './data/output',
            ):
                self.settings = settings
                self.output_dir = output_dir
        """
        self.cfg = cfg_inst
        self.settings = self.cfg.settings
        self.timeset = self.cfg.timeset
        self.output_dir = self.cfg.output_dir

        self.output_var = os.path.join(self.output_dir, 'var')

        OUTPUT_FIG = os.path.join(self.output_dir, 'fig')
        os.makedirs(OUTPUT_FIG, exist_ok=True)
        self.output_fig = OUTPUT_FIG

        str_dt = str(self.settings['Dt'])
        if '.' in str_dt:
            self.dig = len(str_dt.split('.')[1])
        elif 'e' in str_dt:
            self.dig = abs(int(str_dt.split('e')[1]))
        else:
            print('error')


    def animation(self, varname: str, ylim=[-1,1], ext='mp4'):
        """ アニメのプロット """
        N = self.settings['N']
        Dx = self.settings['Dx']
        Dt = self.settings['Dt']
        inittime = self.timeset['plt_inittime']
        timespan = self.timeset['plt_timespan']
        brank = self.timeset['brank']

        fig, ax = plt.subplots(figsize=(6,5), facecolor='w')
        ims = []
        x = np.linspace(0, int(N*Dx)+1, N)
        for time in range(inittime, inittime+timespan+1, brank):
            fpl = np.load(os.path.join(self.output_var, varname, f't={round(time*Dt, self.dig)}.npy'))
            img = ax.plot(x, fpl, c='r') # グラフを作成
            ax.set_ylim(ylim)
            title = ax.text(0.5, 1.01, f'time={round(time*Dt, self.dig)}',
                    ha='center', va='bottom',
                    transform=ax.transAxes, fontsize='large')
        
            ims.append(img+[title]) # グラフを配列に追加 

        # 1000*Dt[ms] ごとに表示
        ani = animation.ArtistAnimation(fig, ims, interval=100)
        ani.save(os.path.join(self.output_fig, f'{varname}.{ext}'))

class plot3d():
    def functz(Upl):
        global X, Y
        z = Upl[X, Y]
        return z

    def render_frame(X, Y, Z, angle):
        """3DグラフをPkLkmageに変換して返す"""
        fig = plt.figure(figsize=(6,5), facecolor='w')
        ax = fig.add_subplot(111, projection='3d')
        # ax = Axes3D(fig)
        ax.plot_wireframe(X, Y, Z, color='r')
        ax.view_init(30, angle+135)
        plt.close()
        # 軸の設定
        ax.set_xlabel('Position')
        ax.set_ylabel('time')
        ax.set_zlabel('Temperature')
        # PIL Image に変換
        buf = BytesIO()
        fig.savefig(buf, bbox_inches='tight', pad_inches=0.0)
        return Image.open(buf)


