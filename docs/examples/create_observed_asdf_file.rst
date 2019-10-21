Creating an ASDF File
=====================

This example demonstrates how the create a new ASDF file from waveform data
in any format ObsPy can read, a QuakeML file, and a list of StationXML files.


.. note::

    Do **NOT** run this with MPI. This would require some modifications and
    is very likely not worth the effort.

.. literalinclude:: create_observed_asdf_file.py
   :language: python
   :linenos:
