Installation
============

NoisePy and Dependencies
------------------------

The nature of NoisePy being composed of python scripts allows flexiable package
installation. What you need to do is essentially build dependented libraries the
scripts and related functions live upon.

``NoisePy`` supports Python version 3.5, 3.6, and 3.7 and it depends on the
following Python modules: ``NumPy``, ``ObsPy``, ``pyasdf``, ``mpi4py``, ``numba``,
``pycwt``. We recommand to use `conda <https://docs.conda.io/en/latest/>`_
and `pip <https://pypi.org/project/pip/>`_ to install the library due to their convinence.
Below are command lines we have tested that would create a python environment to run NoisePy.

.. code-block:: bash
    $ conda create -n noisepy -c conda-forge python=3.7 numpy=1.16.2 numba pandas pycwt jupyter mpi4py=3.0.1 obspy=1.1 pyasdf
    $ conda activate noisepy
    $ git clone https://github.com/mdenolle/NoisePy.git


.. note::
    Please note that the test is performed on *macOS Mojave (10.14.5)*, so it could be slightly different for other OS.


Testing
-------

To assert that your installation is working properly, execute

.. code-block:: bash

    $ python S0A_download_ASDF_MPI.py
    $ python S1_fft_cc_MPI.py
    $ python S2_stacking.py

and make sure the scripts all pass successfully. Otherwise please report issues on the github page or contact the developers.

Github repository of *NoisePy* can be found here: https://github.com/mdenolle/NoisePy
