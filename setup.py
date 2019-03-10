import os
from distutils.core import setup


if __name__ == '__main__':

    with open(os.path.join('qcircuits', 'VERSION'), 'r') as fp:
        version = fp.read().strip()

    setup(
        name='QCircuits',
        version=version,
        author='Andrew M. Webb',
        author_email='andrew@awebb.info',
        packages=['qcircuits'],
        url='http://www.awebb.info/qcircuits/index.html',
        license='LICENSE',
        description='A package for simulating small-scale quantum computing',
        long_description=open('README.rst').read(),
        install_requires=[
            "numpy >= 1.14.3",
        ],
    )
