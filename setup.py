from distutils.core import setup


if __name__ == '__main__':

    setup(
        name='QCircuits',
        version='0.4.2',
        author='Andrew M. Webb',
        author_email='andrew@awebb.info',
        packages=['qcircuits'],
        url='http://www.awebb.info/qcircuits/index.html',
        license='MIT License',
        description='A package for simulating small-scale quantum computing',
        long_description=open('README.rst').read(),
        python_requires='>=3.4.3',
        install_requires=[
            "numpy >= 1.11.3",
        ],
        classifiers=[
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Intended Audience :: End Users/Desktop',
            'Intended Audience :: Developers',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
    )
