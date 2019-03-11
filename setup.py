import setuptools

setuptools.setup(
    name='farms_muscle',
    version='0.1',
    description='Module to generate muscle models',
    url='https://gitlab.com/FARMSIM/farms_muscle.git',
    author='biorob-farms',
    author_email='biorob-farms@groupes.epfl.ch',
    license='MIT',
    packages=setuptools.find_packages(exclude=['tests*']),
    dependency_links=[
        'https://gitlab.com/FARMSIM/farms_muscle.git'],
    install_requires=[
        'numpy',
        'casadi',
        'farms_pylog'
    ],
    zip_safe=False
)
