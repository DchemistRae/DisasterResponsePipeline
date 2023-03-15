# Disaster Response Pipeline Project

This is a data Engineering project, where a data pipeline was used to process messages data from major natural disasters around the world and stored in an SQLite database. Again using NLP (Natural Language Processing) pipelines, the prepared data is read from the SQLite database and classified into categores based on the need communicated by the sender using a machine learning algorithm.

By classifying users text messages sent during a disaster, first respondents will be able to identify quickly the type of incident and provide the appropriate help in that situation.

### Data:
Two csv files were used from Figure Eight (formerly Crowdflower) which crowdsourced the tagging and translation of messages
1. disaster_categories.csv
2. disaster_messages.csv

### File Description

    app
    | - template
    | |- master.html # main page of web app
    | |- go.html # classification result page of web app
    |- run.py # Flask file that runs app
    data
    |- disaster_categories.csv # data to process
    |- disaster_messages.csv # data to process
    |- process_data.py # python ETL file that prepares the data
    |- DisasterResponse.db # database to save clean data to
    |- ETL Pipeline Preparation.ipynb # Jupyter notebook for initial data cleaning
    models
    |- train_classifier.py # NLP file that trains the model 
    |- classifier.pkl # saved model
    |- ML Pipeline Preparation.ipynb # jupyter notebook for initial NLP
    screenshots
    |- web-app1.png # screenshots from the final webapp
    |- web-app2.png
    |- web-app3.png
    |- web-app4.png
    README.md


### Usage:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. once running, open the link http://0.0.0.0:3000/ to access the app

5. Enter text in the field and click classify message to see results.

### Libraries Used

Flask, NLTK, Pandas, Numpy, 
SKlearn, Plotly, Pickle, sqlalchemy

### Screenshots
First Page
![Screenshot - 1](https://github.com/DchemistRae/DisasterResponsePipeline/blob/master/screenshots/web-app1.png)

Second Page
![Screenshot - 1](https://github.com/DchemistRae/DisasterResponsePipeline/blob/master/screenshots/web-app2.png)
