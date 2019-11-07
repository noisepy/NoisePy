Welcome to NoisePy
==================

.. image:: figures/logo_small.png
    :width: 100%
    :align: center


This is the documentation for the Python package of **NoisePy**, which is a new high-performance python tool for seismic ambient noise seismology. For further information and contact information please
see below website:

* Github repository of *NoisePy*: https://github.com/mdenolle/NoisePy

If you use NoisePy for your research and prepare publications, please consider citing **NoisePy**: 

* Jiang, C. and Denolle, M. NoisePy: A new high-performance python tool for seismic ambient noise seismology. In prep for Seismological Research Letter.

The monitoring modules are a compilation of published Python scripts and include the new approach described in:
* Yuan, C., Jiang, C., Bryan, J., Clements, C.T., Okubo, K., Denolle, M.A.: Comparing approaches to measuring time- and frequency-dependent seismic phase variations for coda wave interferometry. In prep for Geophysical Journal International
Please cite this paper if using the monitoring module

We gratefully acknowledge support from the Packard Fundation (`www.packard.org <http://www.packard.org>`_).


Functionality
--------------

* Download continous noise data based on obspy's core functions of `get_station <https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_stations.html>`_ and `get_waveforms <https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_waveforms.html>`_
* Save seismic data in `ASDF <https://asdf-definition.readthedocs.io/en/latest/>`_ format, which convinently assembles meta, wavefrom and auxililary data into one single file (`Turtorials <https://github.com/SeismicData/pyasdf/blob/master/doc/tutorial.rst>`_ on reading/writing ASDF files)
* Offers high flexibility to handle messy SAC/miniSEED data stored on your local machine and convert them into ASDF format data that could easily be pluged into NoisePy
* Performs fast and easy cross-correlation with functionality to run in parallel through `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_
* Includes a series of monitoring functions to measure dv/v on the resulted cross-correlation functions using some recently developed new methods (see our papers for more details) 


.. toctree::
    :hidden:
    :maxdepth: 3
    :glob:

    installation
    tutorial
    examples
    applications

