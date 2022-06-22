![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

# Project: Customer Segmentation/Classification

# Description
The project is about developing a deep learning model to predict the outcome of a marketing campaign based on the collected customer details e.g. age, job, marital status and many more. For the deep learning model, it is developed by training a dense neural network classification model on the [HackerEarth HackLive: Customer Segmentation Dataset](https://www.kaggle.com/datasets/kunalgupta2616/hackerearth-customer-segmentation-hackathon). The raw dataset contains 30,000+ sample of customer details (age, job, education, personal loan) where each customer have been binary-labelled to two categories, whether the customer subscribe or did not subscribe to the focus product of the marketing campaign i.e. Term Deposit.

# How to Install and Run the Project
To run the program on your own device, download/clone the whole repository first. Then, proceed to the directory containing the cloned repository. In this particular directory, locate the `customer_segmentation.py` and run this file in your terminal or any of your favorite IDEs. This will generate all the plots and results especially the trained neural network model.

# Results
## Neural Network Model Plot
![model neural network plot](statics/model.png)

## Model Training/Test Loss and Metrics (Accuracy)
### Matplotlib Plot
![loss metrics plots mpl](statics/perf_vs_epochs.png)
### Smoothed Tensorboard Plot
![loss metrics plots tensoboard](statics/perf_vs_epochs_tensorboard.png)

## Model Performance on The Test/Out-of-Sample Dataset
### Classification Report
![class report](statics/classification_report.png)
### Confusion Matrix
![confusion matrix](statics/confusion_matrix.png)

# Credits
- [HackerEarth HackLive: Customer Segmentation](https://www.kaggle.com/datasets/kunalgupta2616/hackerearth-customer-segmentation-hackathon)
- [Markdown badges source 1](https://github.com/Ileriayo/markdown-badges)
- [Markdown badges source 2](https://github.com/alexandresanlim/Badges4-README.md-Profile)
