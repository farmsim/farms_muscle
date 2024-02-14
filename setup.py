import setuptools
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Compiler import Options
import numpy
import Cython
from farms_container import get_include

Options.docstrings = True
Options.fast_fail = True
Options.annotate = True
Options.warning_errors = False
Options.embedsignature = True
Options.docstrings = True
Options.embed_pos_in_docstring = False
Options.generate_cleanup_code = False
Options.clear_to_none = True
Options.annotate = False
Options.warning_errors = False
Options.error_on_unknown_names = True
Options.error_on_uninitialized = True
Options.convert_range = True
Options.cache_builtins = True
Options.gcc_branch_hints = True
Options.lookup_module_cpdef = False
Options.embed = None
Options.cimport_from_pyx = False
Options.buffer_max_dims = 8
Options.closure_freelist_size = 8

DEBUG = False
extensions = [
    Extension("farms_muscle.muscle",
              ["farms_muscle/muscle.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3', '-lm']
              ),
    Extension("farms_muscle.geyer_muscle",
              ["farms_muscle/geyer_muscle.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3', '-lm']
              ),
    Extension("farms_muscle.millard_rigid_tendon_muscle",
              ["farms_muscle/millard_rigid_tendon_muscle.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3', '-lm']
              ),
    Extension("farms_muscle.millard_damped_equillibrium_muscle",
              ["farms_muscle/millard_damped_equillibrium_muscle.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3', '-lm']
              ),
    Extension("farms_muscle.muscle_system",
              ["farms_muscle/muscle_system.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3', '-lm']
              ),
    Extension("farms_muscle.degroote_muscle",
              ["farms_muscle/degroote_muscle.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3', '-lm']
              ),
    Extension("farms_muscle.physics_interface",
              ["farms_muscle/physics_interface.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3', '-lm']
              ),
    Extension("farms_muscle.bullet_interface",
              ["farms_muscle/bullet_interface.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3', '-lm']
              ),
    Extension("farms_muscle.rigid_tendon",
              ["farms_muscle/rigid_tendon.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3', '-lm']
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
    ext_modules=cythonize(
        extensions, annotate=False,
        compiler_directives={
            # Directives
            'binding': False,
            'embedsignature': True,
            'cdivision': True,
            'language_level': 3,
            'infer_types': True,
            'profile': False,
            'wraparound': False,
            'boundscheck': DEBUG,
            'nonecheck': DEBUG,
            'initializedcheck': DEBUG,
            'overflowcheck': DEBUG,
            'overflowcheck.fold': DEBUG,
            'cdivision_warnings': DEBUG,
            'always_allow_keywords': DEBUG,
            'linetrace': DEBUG,
            # Optimisations
            'optimize.use_switch': True,
            'optimize.unpack_method_calls': True,
            # Warnings
            'warn.undeclared': True,
            'warn.unreachable': True,
            'warn.maybe_uninitialized': True,
            'warn.unused': True,
            'warn.unused_arg': True,
            'warn.unused_result': True,
            'warn.multiple_declarators': True,
        },
    ),
    package_data={
        'farms_muscle': ['*.pxd'],
    },
)
