NoisePy ASDF file architecture
===========================

The ASDF format is developed by the **Theoretical and Computation Seismology Group** at Princeton University, and 
combines the capability to create comprehensive data sets including all necessary meta information with 
high-performance parallel I/O for the most demanding use cases. The users who are interested in the details of this 
format are referred to the following publication.

* Krischer, L., Smith, J., Lei, W., Lefebvre, M., Ruan, Y., de Andrade, E.S., Podhorszki, N., Bozdağ, E. and Tromp, J., 2016. An adaptable seismic data format. Geophysical Supplements to the Monthly Notices of the Royal Astronomical Society, 207(2), 1003-1011.


To better show the pyasdf format, we use the default examples downloaded from the pyasdf Github repository 
https://github.com/SeismicData/pyasdf for creating, processing and writing ``pyasdf`` format data.

* :doc:`examples/create_observed_asdf_file`
* :doc:`examples/process_observed`
* :doc:`examples/parallel_pyflex`

.. toctree::
    :maxdepth: 1
    :hidden:

    examples/create_observed_asdf_file
    examples/process_observed
    examples/parallel_pyflex
    examples/source_receiver_geometry

.. note::
    The examples on pyasdf shown here are exclusively collected from the pyasdf offical website, which is 
    subject to the BSD 3-Clause license.  
