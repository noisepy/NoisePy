Calculate Source-Receiver Geometry
==================================

This simple example demonstrates a fast way to extract the source-receiver
geometry from an ASDF file. It assumes that the ``event_id`` has been correctly
set for each waveform and that these events are part of the global QuakeML
file.

.. literalinclude:: source_receiver_geometry.py
    :language: python
    :linenos:
