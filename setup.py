from setuptools import setup, find_packages

setup(version="1.0",
      name='src',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          'numpy',
          'scipy',
          'numba',
          'obspy',
          'pandas',
          'pyasdf',
          'python>=3.0',
          'mpi4py',
          'markdown',
      ],
      author="Chengxin Jiang & Marine Denolle",
      author_email="chengxin_jiang@fas.harvard.edu & mdenolle@fas.harvard.edu",
      description="A High-performance Computing Python Package for Ambient Noise Analysis",
      license="MIT license",
      url="https://github.com/mdenolle/NoisePy",
      keywords="ambient noise, cross-correlation, seismic monitoring, velocity change "
               " surface wave dispersion",
      platforms='OS Independent',
      classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
      ]
    )
