import PyInstaller.__main__
import os

if __name__ == '__main__':
    sep = os.pathsep
    PyInstaller.__main__.run([
        'app_desktop.py',
        '--name=AstroVisualizer',
        '--windowed',
        f'--add-data=templates{sep}templates',
        f'--add-data=assets{sep}assets',
        '--collect-all=dash',
        '--collect-all=plotly',
        '--collect-all=astroquery',
        '--collect-all=astropy',
        '--collect-all=scipy',
        '--noconfirm',
        '--clean'
    ])
