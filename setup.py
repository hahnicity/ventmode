#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


setup(name='ventmode',
      version="1.0",
      description='Accurate, Reproducible, Ventilator Mode Detection for PB-840 data',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'pandas',
          'prettytable',
          'scipy',
          'scikit-learn',
          'ventmap',
      ],
      entry_points={
      },
      include_package_data=True,
      )
