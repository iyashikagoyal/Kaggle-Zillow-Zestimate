# Kaggle-Zillow-Zestimate 
Kaggle Competition : https://www.kaggle.com/c/zillow-prize-1

Zillow Zestimate

Overview:
The aim is to decrease the log error for the Zillow zestimate using ML model.
Steps followed:
• Performed exploratory data analysis.
• Performed data cleaning, preprocessing, and generated
new features.
• Used the different ML algorithms to get the log error.
• Compared the accuracy and selected the best
approach.
• Tools and library used: Python, Scikit-learn, Numpy,
Pandas, Matplotlib, Plotly
Data Cleaning and Preprocessing:
• Normalized the values.
• Filled the missing values.
• Changed the boolean representation to 0 and 1 by
replacing True with 1 and False with 0.
• Assigned values to NULL in the data.
• Dropped irrelevant features.
• Built new features from the training data.

 • Merged the train_data2016.csv with property_2016.csv file on parcel id.
Training, Validation and Testing:
• Preprocessed data is fed as training data to the model with training and validation split.
• Test data is the merged submission_sample.csv file with the properties_2016.csv on the basis of parcel id.
• Then the dataset is served to different models.
• Best model performing is lightgbm.
Installation
• Install Anaconda.
• Import different libraries: lightgbm, numpy, matplot,
pandas, seaborn, gc.
• Put all the files from the Kaggle in the directory same as
the python file.
• Run the python file on Spyder in Anaconda, or run the
Jupyter notbook.
Output file:
The output file is the csv file: lgb_results_final.csv for the log errors.
The public score is 0.0646684
The private score is 0.0755159

 
