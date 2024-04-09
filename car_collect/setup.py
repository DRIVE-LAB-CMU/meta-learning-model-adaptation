from setuptools import setup

setup(
   name='offroad',
   version='0.0',
   description='A useful module',
   author='Wenli Xiao',
   author_email='',
   packages=['offroad'],  #same as name
   install_requires=[
      'matplotlib',
      'jupyterlab',
      # 'torch',
      'pyproj',
      'ipywidgets',
      # 'gym',
   ], #external packages as dependencies
)