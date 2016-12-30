from distutils.core import setup

setup(
    name='pyslds',
    version='0.0.3',
    author='Matthew J Johnson and Scott W Linderman',
    author_email='mattjj@csail.mit.edu',
    url='https://github.com/mattjj/pyslds',
    packages=['pyslds'],
    install_requires=[
        'numpy', 'scipy', 'matplotlib',
        'pybasicbayes', 'pyhsmm', 'pylds',
        'pypolyagamma>=1.1'
    ])
