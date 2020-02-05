from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(name='helper',
      ext_modules=cythonize([Extension("helper", ["helper.pyx"])],
                            compiler_directives={'language_level':"3"}),
      include_dirs=[numpy.get_include()])