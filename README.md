# airpollution

Data understanding:
The dataset contains information on hourly concentrations of air pollution as well as meteorological data, sourced from the Department of Environment (DOE) Malaysia. The data was collected from an air monitoring station located at Sekolah Menengah Kebangsaan Perempuan Raja Zarina, in Klang, Selangor. The specific coordinates of the monitoring station are Latitude: 3° 00' 62" N and Longitude: 101° 24' 48"E. The data is confidential hence, it cannot be shared publicly.

Step to create the model:
1. Run the data cleaning jupyter notebook
2. Run the model creation jupyter notebook (it will generate 10 models of each air pollution)
3. Paste the model into flask_app/trained_model

Step to run the flask:
1. Activate the environment `venv\Scripts\activate` (if necesscary)
2. Run the script `flask run`

Application interface.
![flask ss](https://user-images.githubusercontent.com/55307820/233702097-ba584beb-95a3-4102-9883-ec02ea06a233.png)
