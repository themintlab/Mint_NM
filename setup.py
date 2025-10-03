from setuptools import setup

setup(
    name='Mint_NM',
    version='0.1.26',
    description='Tools for ENGPHYS 3NM4',
    url='https://github.com/mwelland/themintlab/Mint_NM',
    author='Michael Welland',
    packages=['Mint_NM'],
    install_requires=[
        'numpy',
        'matplotlib',
        'ipywidgets',
        'IPython',
        'pyppeteer',
        'nbconvert',
        'scipy',
        'plotly'
    ],
    python_requires='>=3.7',
    license="CC-BY-4.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
)
