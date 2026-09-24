Mobile Phone Price Analysis & Prediction Using Machine Learning

Project Overview

This project focuses on analyzing mobile phone data and predicting the price of a mobile phone using Machine Learning.

The project follows a complete data science workflow:

Raw Data → Data Cleaning → Data Analysis → Data Visualization → Machine Learning → Price Prediction → Web Application

The prediction is based on mobile phone specifications such as brand, RAM, storage, camera, battery capacity, and screen size.

Objective

The main objective of this project is to develop a Machine Learning model that can predict the approximate price of a mobile phone based on its technical specifications.

Dataset

The project uses a mobile phone dataset containing the following features:

- Brand
- RAM (GB)
- Storage (GB)
- Camera (MP)
- Battery Capacity (mAh)
- Screen Size (inch)
- Price

Dataset Source

The dataset used in this project is included in this repository.

Raw dataset:

"dataset/mobile_raw_data.csv"

Cleaned dataset:

"dataset/mobile_cleaned_data.csv"

Data Cleaning

The raw dataset was cleaned using Python and Pandas.

The cleaning process includes:

- Removing unnecessary spaces from brand names
- Standardizing brand names
- Handling missing numerical values
- Removing duplicate records
- Saving the cleaned dataset as a separate CSV file

Data Analysis

The cleaned dataset was analyzed to understand mobile phone pricing patterns.

The analysis includes:

- Average mobile phone price
- Minimum mobile phone price
- Maximum mobile phone price
- Average price by brand
- Average price by RAM
- Most expensive phone
- Cheapest phone

Data Visualization

Matplotlib was used to create visualizations such as:

- Average Mobile Phone Price by Brand
- RAM vs Mobile Phone Price
- Storage vs Mobile Phone Price

The generated charts help understand the relationship between mobile phone specifications and price.

Machine Learning

A Random Forest Regression algorithm was used for mobile phone price prediction.

Input Features

- Brand
- RAM
- Storage
- Camera
- Battery
- Screen Size

Target Variable

Price

The Brand feature is converted into numerical form using One-Hot Encoding before training the model.

Model Evaluation

The trained model is evaluated using:

- Mean Absolute Error (MAE)
- R² Score

These metrics are used to measure the prediction performance of the regression model.

Web Application

A Flask-based web application is included in the project.

Users can enter mobile phone specifications through a web form and receive an estimated phone price.

Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Flask
- Joblib
- VS Code

Project Structure

Mobile-Phone-Price-Prediction/
│
├── AnkitaDey_MobilePhonePricePrediction.py
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── mobile_raw_data.csv
│   └── mobile_cleaned_data.csv
│
├── model/
│   └── phone_price_prediction_model.pkl
│
└── templates/
    └── index.html

Installation

Make sure Python 3.11 or a compatible Python version is installed.

Install the required libraries using:

pip install -r requirements.txt

How to Run the Project

Step 1: Run the main Python project

Open the project folder in VS Code and run:

python AnkitaDey_MobilePhonePricePrediction.py

This will:

- Load the raw dataset
- Clean the data
- Save the cleaned dataset
- Perform data analysis
- Generate visualizations
- Train the Machine Learning model
- Evaluate the model
- Save the trained model
- Perform a sample price prediction

Step 2: Run the Flask Web Application

Run:

python app.py

Then open the following address in a web browser:

http://127.0.0.1:5000

Enter the mobile phone specifications and click Predict Phone Price.

Key Information

Project Title: Mobile Phone Price Analysis & Prediction Using Machine Learning

Student Name: Ankita Dey

Project Type: Data Analytics and Machine Learning

Machine Learning Algorithm: Random Forest Regression

Target Variable: Mobile Phone Price

Development Environment: Visual Studio Code

Conclusion

This project demonstrates a complete Machine Learning workflow, starting from raw and unclean data and progressing through data cleaning, analysis, visualization, model training, evaluation, and price prediction.

The project also demonstrates how a Machine Learning model can be integrated into a Flask web application for interactive predictions.