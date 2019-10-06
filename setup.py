from setuptools import setup

setup(
    name='psp', 
    packages=['psp'],
    version='0.0.1.dev',
    entry_points={
        'console_scripts': ['psp-cli=psp.cmd:main']
    }
)