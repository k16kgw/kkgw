from setuptools import setup

setup(
    name = 'kkgw',
    version = '1.1.1',
    url = 'https://github.com/keikagawa/kkgw.git',
    license = 'Free',
    author = 'keikagawa',
    author_email = 'gerogero7429@gmail.com',
    description = 'My function storage',
    install_requires = ['setuptools'],
    packages = ["kkgw", "kkgw.utils", "kkgw.math"],
    # entry_points = {
    #     'console_scripts': [
    #         'top = kkgw.top:main',
    #         'bottom = kkgw.utils.bottom:sub'
    #     ]
    # }
)