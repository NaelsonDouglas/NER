from setuptools import setup

setup(
    name='ner',
    version='0.0.1',
    description='',
    author='naelsondouglas',
    author_email='naelson17@gmail.com',
    packages=[],
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'spacy',
        'fastparquet',
        'pyarrow',
        'rich',
        'tqdm',
        'torch',
        'sklearn',
        'cupy-cuda11x',
        'spacy-transformers',
        'thefuzz',
        'fastapi'
    ],
    scripts=[],
)