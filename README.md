# Data_Scientist_Disaster-Response-Pipeline

This project is part of my Data Sciene Nanodegree project by Udacity. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Go to `app` directory: `cd app`
3. Run your web app: `python run.py`
4. Click the `PREVIEW` button to open the homepage

## Libraries
The developer needs to import the following libraries to run the analysis:
- sys
- numpy 
- pandas 
- sqlalchemy
- re
- nltk
- pickle
- sklearn
- json
- plotly
- flask

## Motivation for project
In emergency situations and disasters (e.g., flood events, hurricanes, earthquakes, terrorist attacks) information is key. However, sometimes the amount of information is overwhelming and it is hard to filter and categorize. In order to quickly react on disaster messages, machine learning algorithms are capable of supporting the receiver. By means of this project I am trying to contribute to that issue!
 
## Dataset
In this project, I use disaster data provided by [Figure Eight](https://www.figure-eight.com/) to train the model. The data contain real messages that were sent during disaster events!

## Result
With this project, I created a machine learning pipeline to categorize disaster events so that the user can send the messages to an appropriate disaster relief agency.

![plot]([https://github.com/nikextens/Operationalizing_Machine_Learning/blob/main/screen-shot-2020-09-15-at-12.36.11-pm.png](https://github.com/nikextens/Data_Scientist_Disaster-Response-Pipeline/blob/main/TestData.png)](https://github.com/nikextens/Data_Scientist_Disaster-Response-Pipeline/blob/main/LandingPage.png))

My project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data. 

![plot]([https://github.com/nikextens/Operationalizing_Machine_Learning/blob/main/screen-shot-2020-09-15-at-12.36.11-pm.png](https://github.com/nikextens/Data_Scientist_Disaster-Response-Pipeline/blob/main/TestData.png))

## Acknowledgments
Special thanks go to Udacity and Figure Eight which provided the dataset and useful hints to master this challenging task!





