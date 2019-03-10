from distutils.core import setup


if __name__ == '__main__':

    setup(
        name='QCircuits',
        version='0.2.0',
        author='Andrew M. Webb',
        author_email='andrew@awebb.info',
        packages=['qcircuits'],
        url='http://www.awebb.info/qcircuits/index.html',
        license='MIT',
        description='A package for simulating small-scale quantum computing',
        long_description=open('README.rst').read(),
        install_requires=[
            "numpy >= 1.11.3",
        ],
    )
