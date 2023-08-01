# Air Quality and weather open data: application of ML methods

This paper aims to define a ML model capable of making valid predictions of the air pollutant *Nitrogen Dioxide* (NO2) after 24 hours for three stations of the *ARPAE* air quality monitoring network of the province of Rimini, exploiting the historical data of the pollutant and the weather data. The data has been collected and manipulated to have an optimal structure to be batched and fed to *machine learning* algorithms. Different neural network structures are compared to choose the one that provides the most accurate predictions. Finally, results of the various algorithms and some considerations are shown.

The code is divided in:
- *0_air_data.py*: reorganizes air quality data, from raw to dataframes;
- *1_meteo_data.py*: reorganizes weather data, from raw to dataframes;
- *2_air_meteo_data.py*: merges the selected air quality and weather data;
- *3_ANN.py*: loads the selected and structured data, returns the results of the choosen neural network.

Each of these files can be run independently, since the intermediate files are in the repository.

The complete report is contained in the *report* directory.