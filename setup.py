from setuptools import setup
from setuptools import find_packages

setup(
    name='odeopt',
    version='0.0.0',
    description='Optimization based solve for ODE parameter inference problem.',
    url='https://github.com/ihmeuw-msca/ODEOPT',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'pytest',
    ],
    zip_safe=False,
)
