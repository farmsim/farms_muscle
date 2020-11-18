import setuptools
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Compiler import Options
import numpy
import Cython

Options.docstrings = True
Options.fast_fail = True
Options.annotate = True
Options.warning_errors = False

extensions = [
    Extension("farms_muscle.muscle",
              ["farms_muscle/muscle.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_muscle.geyer_muscle",
              ["farms_muscle/geyer_muscle.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_muscle.millard_rigid_tendon_muscle",
              ["farms_muscle/millard_rigid_tendon_muscle.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_muscle.millard_damped_equillibrium_muscle",
              ["farms_muscle/millard_damped_equillibrium_muscle.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_muscle.muscle_system",
              ["farms_muscle/muscle_system.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_muscle.degroote_muscle",
              ["farms_muscle/degroote_muscle.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_muscle.physics_interface",
              ["farms_muscle/physics_interface.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_muscle.bullet_interface",
              ["farms_muscle/bullet_interface.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              )
]

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
        "https://gitlab.com/FARMSIM/farms_pylog.git"],
    install_requires=['Cython',
                      'pyyaml',
                      'numpy',
                      'scipy',
                      'farms_pylog'],
    zip_safe=False,
    ext_modules=cythonize(extensions, annotate=True),
    package_data={
        'farms_muscle': ['*.pxd'],
    },
)
