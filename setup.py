

from setuptools import setup

setup(
   name='paraflow',
   version='1.0',
   description='the open source parametric passage flow generator',
   author='Afshawn Lotfi',
   author_email='',
   packages=['paraflow'],
   install_requires=[
    "numpy",
    "scipy",
    "ezmesh @ git+https://github.com/Turbodesigner/ezmesh.git",
    "mpi4py"
   ]
)