# Disaster Response Pipeline Project

This is a data Engineering project, where a data pipeline was used to process messages data from major natural disasters around the world and stored in an SQLite database. Again using NLP (Natural Language Processing) pipelines, the prepared data is read from the SQLite database and classified into categores based on the need communicated by the sender using a machine learning algorith.

### Data:
Two csv files were used from Figure Eight (formerly Crowdflower) which crowdsourced the tagging and translation of messages
1. disaster_categories.csv
2. disaster_messages.csv

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
