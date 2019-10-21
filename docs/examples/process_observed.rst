Processing Observed Data in Parallel
====================================

This fairly complex examples takes an ASDF file and produces two new data
sets, each processed in a different frequency band.

It can be run with MPI. It scales fairly well and will utilize parallel I/O if
your machine supports it. Please keep in mind that there is a significant
start-up cost for Python on each core (special Python versions that get around
that if really necessary are in existence) so don't use too many cores.

.. code-block:: bash

    $ mpirun -n 64 python process_observed.py


If you don't run it with MPI with will utilize Python's ``multiprocessing``
module and run it on each of the machines cores. I/O is not parallel and
uses a round-robin scheme where only one core writes at single point in time.


.. code-block:: bash

    $ python process_observed.py


.. literalinclude:: process_observed.py
    :language: python
    :linenos:
