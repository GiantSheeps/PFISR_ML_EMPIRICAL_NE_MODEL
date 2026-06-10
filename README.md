# PFISR Electron Density Forecasting Pipeline

This repository contains the machine learning architecture used to forecast upper atmospheric electron density (Ne) profiles. The pipeline trains models on hourly binned data from the Poker Flat Incoherent Scatter Radar (PFISR) between 100 and 400 km altitudes. It specifically evaluates the predictive influence of geomagnetic and solar drivers using a strict out-of-sample validation framework.

The core algorithms include Random Forest, Gradient Boosting, Multilayer Perceptron, and a baseline Linear Regression model. 

## Getting the Data

To keep the repository lightweight, the primary training dataset is hosted externally on Zenodo. 

1. Download `clean_pfisr_data.nc` from the [Zenodo repository](https://zenodo.org/records/20631519?preview=1&token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc4MTExMzg4NiwiZXhwIjoxODg2Mjg0Nzk5fQ.eyJpZCI6IjI1NDJmMDIyLWIyZjctNDBlYy04YTVhLWVhYmI4ODhlOTM1MyIsImRhdGEiOnt9LCJyYW5kb20iOiJkNWZjMmJjYjhjZDY1OGYxNDNlOGZiY2NmNGNmYTk2YSJ9.RsOohZKOnk8L7wT5tdJCg7QjWiD49Om9S2-XytfRAXf9M4QTcwKwtBx4Sk8waXjq2H0k1GjpaiUEK8R3CEGW8A).
2. Place the downloaded NetCDF file inside the `data/` directory at the root of this project.

## Repository Structure

* `models.py`: Contains the class definitions for the tabular models and the scikit-learn randomized search distribution configurations.
* `example_notebook.ipynb`: The main interactive Jupyter notebook. This handles data ingestion, error thresholding, hyperparameter tuning, model training, and out-of-sample permutation importance testing.

## Running the Code

Ensure your Python environment has the necessary dependencies installed (`numpy`, `pandas`, `xarray`, `scikit-learn`, `joblib`, and `scipy`). 

Open `example_notebook.ipynb` and run the cells sequentially. The first section will automatically train the models and generate a local `ckpts/` folder to store the weights. The second section will rehydrate those weights to evaluate the models against unseen validation years and output the final metrics to a local CSV.
