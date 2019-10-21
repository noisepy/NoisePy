
# Application scripts of NoisePy
The NoisePy provides two additional scripts located in `src/application_scripts` to 1) extract dispersion information and 2) measure dv/v through time using the resulted cross-correlation functions of the NoisePy package. Below is a short tutorial on how the two scripts work.

**Surface wave dispersion analysis using `I_group_velocity.py`**\

To show how this script works, we use the default setting of the downloading script (S0A) of NoisePy to download 1 month of continous noise data of the CI stations for 2016 Jun, and compute the cross-correlation using a `cc_len` of 30 min and `step` of 7.5 min. Then we stack the daily cross-correlation functions from these small time window before finally stacking them all together to form a montly-stacked cross-correlation. During the stacking, we choose to keep both the `linear` and `pws` stacked results. After running through S0A, S1 and S2 scripts, we get a foler of `STACK` in our `rootpath` directory, where it contains all stacked cross-correlation with many sub-folders inside the folder representing each virual source.

In order to extract the surface wave dispersion information, we change the parameters of xxx to our resulted STACK data, xxx. We set `ncomp` to be 3 to check the dispersion image on multiple components. After setting the parameters, we run:

```python
python I_group_velocity.py
```  

The script would then generate a PDF file showing all 9 cross-component dispersive image in the specified fold along with three txt files containing the extracted dispersion curves for `ZZ`, `RR` and `TT` components. The snapshort below shows how the dispersive image should look like! Note that, using different wavelet parameters would slightly change the resulted dispersive images, but should not much. Here we only use the default values by the [pycwt](https://pycwt.readthedocs.io/en/latest/) module. 
<img src="/docs/src/dispersion_image.png" width="700" height="520">

We compare the resulted dispersion curves from this script with those extracted using Frequency-Time analysis, and they are very similar to each other. However, we have to note that we do not correct for the frequency effects. According to Bensen et al., (2007), the centrel frequncy of each narrowband filters (equvalent to wavelet tranforms at each scale)would be different from the instaneous frequency calculated using instaneous. This difference tend to be significant when the amplitude spectral is not flat and cause spectral linkage. This is going to be included in a future release. 


**Monitoring application using `II_measure_dvv.py`**\
To run this script, we use the default setting of the downloading script (S0A) of NoisePy to download 1 month of continous noise data of the CI stations for 2016 Jun, and compute the cross-correlation using a `cc_len` of 30 min and `step` of 7.5 min. Then we stack the daily cross-correlation functions from these small time window before finally stacking them all together to form a montly-stacked cross-correlation.
