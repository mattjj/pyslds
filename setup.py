from distutils.core import setup

setup(
    name='pyslds',
    version='0.0.1',
    author='Matthew James Johnson',
    author_email='mattjj@csail.mit.edu',
    url='https://github.com/mattjj/pyhsmm-slds',
    packages=['pyslds'],
    install_requires=[
        'numpy', 'scipy', 'matplotlib',
        'pybasicbayes', 'pyhsmm', 'pylds',
    ])

