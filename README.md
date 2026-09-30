# ATM Failure Prediction
## Project Overview

ATM Failure Prediction is a Machine Learning project developed to predict whether an ATM is likely to fail within the next 48 hours. The project uses historical ATM operational, technical, maintenance, and customer-related data to identify patterns associated with ATM failures.

The prediction can help organizations identify potentially problematic ATMs in advance and take preventive maintenance actions.

## Problem Statement

ATM failures can interrupt banking services and cause inconvenience to customers. Unexpected failures may be related to factors such as transaction activity, hardware errors, cash availability, network issues, power outages, maintenance history, and previous failures.

The objective of this project is to build a machine learning classification model that predicts the target variable Failure_Next_48H, indicating whether an ATM is expected to fail within the next 48 hours.


## Target Variable

Failure_Next_48H
- `0` – ATM is not expected to fail within the next 48 hours
- `1` – ATM is expected to fail within the next 48 hours


### Machine Learning Model

A Random Forest Classifier was used to predict whether an ATM will fail within the next 48 hours.

Random Forest was selected because it can handle multiple numerical and categorical-derived features and is capable of capturing non-linear relationships between ATM conditions and failure events.


### Conclusion

This project demonstrates how machine learning can be used for predictive maintenance in ATM systems. By analysing operational, technical, network, power, and maintenance-related information, the system can identify ATMs that may be at risk of failure within the next 48 hours.

The Streamlit application provides an easy-to-use interface for making predictions and understanding the analysis behind the model.
