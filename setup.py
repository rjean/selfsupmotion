from setuptools import setup, find_packages


setup(
    name='selfsupmotion',
    version='0.0.1',
    packages=find_packages(include=['selfsupmotion', 'selfsupmotion.*']),
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'main=selfsupmotion.main:main'
        ],
    }
)
