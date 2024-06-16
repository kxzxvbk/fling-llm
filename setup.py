# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module setuptools script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from importlib import import_module
from setuptools import setup, find_packages

meta_module = import_module('fling_llm')
meta = meta_module.__dict__
here = os.path.abspath(os.path.dirname(__file__))
with open('README.md', mode='r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name=meta['__TITLE__'],
    version=meta['__VERSION__'],
    description=meta['__DESCRIPTION__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    author=meta['__AUTHOR__'],
    author_email=meta['__AUTHOR_EMAIL__'],
    url='https://github.com/kxzxvbk/fling-llm',
    license='Apache License, Version 2.0',
    keywords='Federated Learning, Large Language Model',
    packages=[
        *find_packages(include=('fling_llm', 'fling_llm.*')),
        *find_packages(include=('zoo', 'zoo.*')),
    ],
    package_data={
        package_name: ['*.yaml', '*.xml', '*cfg', '*SC2Map']
        for package_name in find_packages(include='fling.*')
    },
    python_requires=">=3.7",
    install_requires=[
        'Fling>=0.0.4', 'transformers', 'datasets', 'accelerate'
    ],
    extras_require={
        'test': [
            'coverage>=5,<=7.0.1',
            'mock>=4.0.3',
            'pytest~=7.0.1',  # required by gym>=0.25.0
            'pytest-cov~=3.0.0',
            'pytest-mock~=3.6.1',
            'pytest-xdist>=1.34.0',
            'pytest-rerunfailures~=10.2',
            'pytest-timeout~=2.0.2'
        ],
        'style': [
            'yapf==0.29.0',
            'flake8<=3.9.2',
            'importlib-metadata<5.0.0',  # compatibility
        ]
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
