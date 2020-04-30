#!/usr/bin/env python

from setuptools import setup
from setuptools.command.test import test as TestCommand

class NoseTestCommand(TestCommand):

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import nose
        nose.run_exit(argv=['nosetests'])

setup(name='Rhodium',
      version='1.0',
      description='Robust decision making and exploratory modelling in Python',
      author='David Hadka',
      author_email='dhadka@users.noreply.github.com',
      license="GNU GPL version 3",
      url='https://github.com/Project-Platypus/Rhodium',
      packages=['rhodium'],
      install_requires=['prim', 'seaborn', 'mpldatacursor', 'pydot', 'SALib'],
      tests_require=['nose', 'mock'],
      cmdclass={'test': NoseTestCommand},
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Education',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Libraries :: Python Modules',
          ],
     )
