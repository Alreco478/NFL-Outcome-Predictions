from setuptools import setup, find_packages

setup(
    name='data_muncher',
    version=1.0,
    author='Alexander Coover',
    url = 'https://github.com/Alreco478/DAT-Capstone',
    packages=find_packages(),
    install_requires=[
        'nfl_data_py'
    ]
    
)
