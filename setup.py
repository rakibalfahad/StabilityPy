import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

# Ensure numpy and scipy are in the install_requires
if 'numpy' not in INSTALL_REQUIRES:
    INSTALL_REQUIRES.append('numpy>=1.20.0')
if 'scipy' not in INSTALL_REQUIRES:
    INSTALL_REQUIRES.append('scipy>=1.7.0')

# Development dependencies
DEV_REQUIRES = [
    "black",
    "flake8",
    "isort",
    "pre-commit",
    "pytest>=7.0.0",
    "pytest-cov>=4.1.0",
    "pycodestyle",
]

# GPU acceleration dependencies
GPU_REQUIRES = [
    "torch>=1.7.0",
]

setup(
    name='stability-selection',
    version='0.1.0',
    description='A scikit-learn compatible implementation of stability selection for feature selection with GPU acceleration',
    author='Updated from Thomas Huijskens original',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    setup_requires=['numpy>=1.20.0', 'scipy>=1.7.0'],  # Required during setup
    extras_require={
        'dev': DEV_REQUIRES,
        'gpu': GPU_REQUIRES,
        'all': DEV_REQUIRES + GPU_REQUIRES,
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)