# DESCRIPTION
Observation-based data reconstructions of global surface ocean CO2 fugacity (fuCO2) play an essential role in
monitoring the recent status of ocean carbon uptake and ocean acidification and their impacts on marine organisms and ecosystems. One key resource supporting fuCO2 reconstruction is the Surface Ocean CO2 Atlas (SOCAT, https://socat.info/) database. Despite providing millions of data in the last decades, SOCAT has covered a tiny portion (2%) of the global ocean and the periods of interest. Besides, the SOCAT’s annual release usually lags behind the real time (up to 1.5 years) that obstructs timely quantification of recent variations of carbon fluxes between the Earth System components, not only with the ocean.

This Python source code exemplifies a feed-forward neural network (FFNN) model (**FFNN_model.py**). Within the construction of an ensemble approach of FFNN models, namely **CMEMS-LSCE-FFNN**, it was developed at *Laboratoire des Sciences du Climat et de l’Environnement (LSCE)* under the *Copernicus Marine Environment Monitoring Service (CMEMS) project* for sustainably reconstructing observation-derived global ocean carbonate system datasets at high space-time resolutions since the year 1985 (Chau et al., 2022, 2023a). 

The FFNN model provided here is trained on monthly, 1° - gridded data of fuCO2 and a suite of environmental variables (e.g., sea surface temperature - SST, salinity - SSS, Chlorophyll - CHL, etc.) in May to July over the period 1985-2021. However, data extrapolations to a finer resolution and forward in time are doable (Chau et al., 2023a,b). Any implementation of the Python script code is encouraged and subject to citing the following references.


# REFERENCES:
1. Chau T T T, Gehlen M, Chevallier F (2022). “A seamless ensemble-based reconstruction of surface ocean pCO2 and air–sea CO2 fluxes over the global coastal and open oceans", Biogeosciences, 19, 1087–1109, DOI:10.5194/bg-19-1087-2022.
2. Chau T T T, Gehlen M, Metzl N and Chevallier F (2023a). “CMEMS-LSCE: A global 0.25-degree, monthly reconstruction of the surface ocean carbonate system", Earth System Data, DOI: 10.5194/essd-2023-146.
3. Chau T T T, Chevallier F and Gehlen M (2023b). “Global analysis of surface ocean CO2 fugacity and air-sea fluxes with low latency", ESS Open Archive, DOI: 10.22541/essoar.169711695.54822708/v1, submitted to the AGU journal - Geophysical Research Letters.


# CONTACT:
Authors: Thi-Tuyet-Trang Chau (thi.tuyet.trang.chau@gmail.com)
