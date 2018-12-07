# Copyright 2017-2018 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Install script for setuptools."""

import os
import platform
import subprocess
import sys

from distutils import cmd
from distutils import log
from setuptools import find_packages
from setuptools import setup
from setuptools.command import install
from setuptools.command import test

PLATFORM_SUFFIXES = {
    'Linux': 'linux',
    'Windows': 'win64',
    'Darwin': 'macos',
}
DEFAULT_HEADERS_DIR = '~/.mujoco/mujoco200_{}/include'.format(
    PLATFORM_SUFFIXES[platform.system()])

# Relative paths to the binding generator script and the output directory.
AUTOWRAP_PATH = 'dm_control/autowrap/autowrap.py'
MJBINDINGS_DIR = 'dm_control/mujoco/wrapper/mjbindings'

# We specify the header filenames explicitly rather than listing the contents
# of the `HEADERS_DIR` at runtime, since it will probably contain other stuff
# (e.g. `glfw.h`).
HEADER_FILENAMES = [
    'mjdata.h',
    'mjmodel.h',
    'mjrender.h',
    'mjui.h',
    'mjvisualize.h',
    'mjxmacro.h',
    'mujoco.h',
]


def _initialize_mjbindings_options(cmd_instance):
  """Set default values for options relating to `build_mjbindings`."""
  # A default value must be assigned to each user option here.
  cmd_instance.inplace = 0
  cmd_instance.headers_dir = os.path.expanduser(DEFAULT_HEADERS_DIR)


def _finalize_mjbindings_options(cmd_instance):
  """Post-process options relating to `build_mjbindings`."""
  header_paths = []
  for filename in HEADER_FILENAMES:
    full_path = os.path.join(cmd_instance.headers_dir, filename)
    if not os.path.exists(full_path):
      raise IOError('Header file {!r} does not exist.'.format(full_path))
    header_paths.append(full_path)
  cmd_instance.header_paths = ' '.join(header_paths)


class BuildMJBindingsCommand(cmd.Command):
  """Runs `autowrap.py` to generate the low-level ctypes bindings for MuJoCo."""
  description = __doc__
  user_options = [
      # The format is (long option, short option, description).
      ('headers-dir=', None,
       'Path to directory containing MuJoCo headers.'),
      ('inplace=', None,
       'Place generated files in source directory rather than `build-lib`.'),
  ]
  boolean_options = ['inplace']

  def initialize_options(self):
    _initialize_mjbindings_options(self)

  def finalize_options(self):
    _finalize_mjbindings_options(self)

  def run(self):
    cwd = os.path.realpath(os.curdir)
    if self.inplace:
      dist_root = cwd
    else:
      build_cmd = self.get_finalized_command('build')
      dist_root = os.path.realpath(build_cmd.build_lib)
    output_dir = os.path.join(dist_root, MJBINDINGS_DIR)
    command = [
        sys.executable or 'python',
        AUTOWRAP_PATH,
        '--header_paths={}'.format(self.header_paths),
        '--output_dir={}'.format(output_dir)
    ]
    self.announce('Running command: {}'.format(command), level=log.DEBUG)
    try:
      # Prepend the current directory to $PYTHONPATH so that internal imports
      # in `autowrap` can succeed before we've installed anything.
      old_environ = os.environ.copy()
      new_pythonpath = [cwd]
      if 'PYTHONPATH' in old_environ:
        new_pythonpath.append(old_environ['PYTHONPATH'])
      os.environ['PYTHONPATH'] = ':'.join(new_pythonpath)
      subprocess.check_call(command)
    finally:
      os.environ = old_environ


class InstallCommand(install.install):
  """Runs 'build_mjbindings' before installation."""

  user_options = (
      install.install.user_options + BuildMJBindingsCommand.user_options)
  boolean_options = (
      install.install.boolean_options + BuildMJBindingsCommand.boolean_options)

  def initialize_options(self):
    install.install.initialize_options(self)
    _initialize_mjbindings_options(self)

  def finalize_options(self):
    install.install.finalize_options(self)
    _finalize_mjbindings_options(self)

  def run(self):
    self.reinitialize_command('build_mjbindings',
                              inplace=self.inplace,
                              headers_dir=self.headers_dir)
    self.run_command('build_mjbindings')
    install.install.run(self)


class TestCommand(test.test):
  """Prepends path to generated sources before running unit tests."""

  def run(self):
    # Generate ctypes bindings in-place so that they can be imported in tests.
    self.reinitialize_command('build_mjbindings', inplace=1)
    self.run_command('build_mjbindings')
    test.test.run(self)

setup(
    name='dm_control',
    description='Continuous control environments and MuJoCo Python bindings.',
    author='DeepMind',
    license='Apache License, Version 2.0',
    keywords='machine learning control physics MuJoCo AI',
    install_requires=[
        'absl-py',
        'enum34',
        'future',
        'futures',
        'glfw',
        'lxml',
        'numpy',
        'pyopengl',
        'pyparsing',
        'setuptools',
        'six',
    ],
    tests_require=[
        'mock',
        'nose',
        'pillow',
        'scipy',
    ],
    test_suite='nose.collector',
    packages=find_packages(),
    package_data={
        'dm_control.composer': [
            'arena.xml'
        ],
        'dm_control.mjcf': [
            'schema.xml',
            'test_assets/*.xml',
            'test_assets/meshes/*.stl',
            'test_assets/meshes/more_meshes/*.stl',
            'test_assets/textures/*.png',
        ],
        'dm_control.mujoco.testing': [
            'assets/*.png',
            'assets/*.stl',
            'assets/*.xml'
            'assets/frames/*.png',
        ],
        'dm_control.suite': [
            '*.xml',
            'common/*.xml'
        ],
    },
    cmdclass={
        'build_mjbindings': BuildMJBindingsCommand,
        'install': InstallCommand,
        'test': TestCommand,
    },
    entry_points={},
)
