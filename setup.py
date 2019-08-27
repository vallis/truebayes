#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

requirements = []
test_requirements = []

setup(
    name = 'truebayes',
    version = '0.1',
    description = "TrueBayes",
    long_description = "Learning Bayes theorem with a neural network for gravitational-wave inference",

    author = "M. Vallisneri and A. Chua",
    author_email = 'vallis@vallis.org',
    url = 'https://github.com/vallis/truebayes',

    packages = ['truebayes'],
    package_dir = {'enterprise': 'enterprise'},
    include_package_data = True,
    package_data = {'enterprise': ['data/*']},
    install_requires = requirements,

    license = "MIT license",
    zip_safe = False,
    keywords = 'enterprise',
    classifiers = ['Development Status :: 2 - Pre-Alpha',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: MIT License',
                   'Natural Language :: English',
                   'Programming Language :: Python :: 3.7',
    ],

    test_suite = 'tests',
    tests_require = test_requirements
)
