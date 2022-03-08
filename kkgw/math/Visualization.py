import os

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt 
import matplotlib.pylab as p
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import animation, rc

from io import BytesIO
from PIL import Image

import pickle

def close_fig(fig, close):
    if close:
        plt.close(fig)
    else:
        plt.show()

class Plot2d():
    def __init__(self, output_dir):
        self.output_dir = output_dir
        OUTPUT_FIG = output_dir / 'fig'
        OUTPUT_FIG.mkdir(parents=True, exist_ok=True)
        self.output_fig = OUTPUT_FIG

    def snapshot_sol(self, Upl, time, close=True):
        """ 各時刻での空間x未知関数のグラフ """
        OUTPUT_FIG_U = self.output_fig / 'U'
        OUTPUT_FIG_U.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(6,5), facecolor='w')
        ax = fig.add_subplot(
            title=f'time={time}',
            xlabel='Position',
            ylabel='Solution'
        )
        x = list(range(0, Upl.size))
        ax.plot(x, Upl)
        fig.savefig(os.path.join(OUTPUT_FIG_U, f't={time}.png'))
        close_fig(fig, close)

    def timeseries(self, x, title: str, inittime: int, timespan: int, Dt, close=True):
        fig = plt.figure(figsize=(6,5), facecolor='w')
        ax = fig.add_subplot(111)
        t = np.linspace(inittime, int(timespan*Dt)+1, timespan+1)
        plt.plot(t, x, color='r')
        ax.set_xlabel('time')
        ax.set_ylabel(title)
        ax.set_title(title)
        fig.savefig(os.path.join(self.output_fig, f'{title}.png'))
        close_fig(fig, close)

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


