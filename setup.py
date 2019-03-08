from distutils.core import setup

setup(
    name='QCircuits',
    version='0.1.0',
    author='Andrew M. Webb',
    author_email='andrew@awebb.info',
    packages=['qcircuits'],
    url='http://pypi.python.org/pypi/QCircuits/',
    license='LICENSE',
    description='A package for simulating small-scale quantum computing',
    long_description=open('README.rst').read(),
    install_requires=[
        "numpy >= 1.14.3",
    ],
)
