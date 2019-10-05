from setuptools import setup

setup(
    name='pfp', 
    packages=['pfp'],
    version='0.0.1.dev1',
    entry_points={
        'console_scripts': ['pfp-cli=pfp.cmd:main']
    }
)