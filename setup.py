"""

A setuptools based setup module
with single-source versioning

See:
https://packaging.python.org/en/latest/distributing.html
https://packaging.python.org/guides/single-sourcing-package-version/

"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
import re

here = path.abspath(path.dirname(__file__))

def read(*parts):
    filename = path.join(here, *parts)
    with open(filename, encoding='utf-8') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name='neural_rag',
    version=find_version('nn_rag', '__init__.py'),
    description='',
    long_description=read('README.md'),
    url='http://github.com/gigas64/hadron-nn',
    author='Gigas64',
    author_email='gigas64@aistac.net',
    classifiers=[
        # 'Development Status :: 1 - Planning',
        # 'Development Status :: 2 - Pre-Alpha',
        'Development Status :: 3 - Alpha',
        # 'Development Status :: 4 - Beta',
        # 'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Adaptive Technologies',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='',
    packages=find_packages(exclude=['tests']),
    license='MIT',
    include_package_data=True,
    package_data={},
    install_requires=[
        'pyarrow',
        'pandas',
        'numpy',
        'torch',
        'tensorflow',
        'matplotlib',
        'seaborn',
        'scipy',
        'scikit-learn',
        'requests',
        'pymupdf',
        'pymupdf4llm',
        'sentence_transformers',
        'spacy',
        'tqdm',
        'transformers',
        'accelerate',
        'bitsandbytes',
        'wheel',
    ],
    extras_require={},
    test_suite='tests',
)
