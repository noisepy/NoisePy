Running pyflex in Parallel
==========================

``pyasdf`` can be used to run a function across the data from two ASDF data
sets. In most cases it will be some kind of misfit or comparision function.
This example runs `pyflex <http://krischer.github.io/pyflex/>`_ to pick
windows given a data set of observed and another data set of synthetic data.

It can only be run with MPI:

.. code-block:: bash

    $ mpirun -n 16 python parallel_pyflex.py


.. literalinclude:: parallel_pyflex.py
    :language: python
    :linenos:
