

from setuptools import setup

setup(
   name='paraflow',
   version='3.5.0',
   description='the open source parametric passage flow generator',
   author='Afshawn Lotfi',
   author_email='',
   packages=['paraflow', 'paraflow.passages', 'paraflow.simulation'],
   install_requires=[
    "numpy",
    "scipy",
    "ezmesh @ git+https://github.com/OpenOrion/ezmesh.git@3.6.5",
    "pymoo",
    "thermo",
    "matplotlib",
    "plotly",
    "vtk"
   ]
)
