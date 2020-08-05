# coding: utf-8
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import os
import sysconfig


def get_ext_filename_without_platform_suffix(filename):
    name, ext = os.path.splitext(filename)
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix == ext:
        return filename
    ext_suffix = ext_suffix.replace(ext, '')
    idx = name.find(ext_suffix)
    if idx == -1:
        return filename
    else:
        return name[:idx] + ext


class BuildExtWithoutPlatformSuffix(build_ext):
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        return get_ext_filename_without_platform_suffix(filename)


setup(name='helper',
      ext_modules=cythonize([Extension("random_walk", ["random_walk.pyx"])],
                            compiler_directives={'language_level': "3"}),
      include_dirs=[numpy.get_include()],
      cmdclass={'build_ext': BuildExtWithoutPlatformSuffix})
