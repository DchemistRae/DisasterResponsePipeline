import sys
# import libraries
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    '''
    load data from SQLite database and split into predictors and target variable

    input
    database_filepath   filepath to database containing training data

    Returns
    X   predictor variable or messages
    Y   Target variable or categories
    category_name   column names of target variables
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('processed_data', engine)
    Y = df.iloc[:,4:]
    X = df['message']
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text):
    '''
    takes in text messages, apply a number of text processing to them and return word tokens formed form the text

    input
    text   input message to be normalized & tokenized

    Returns
    clean_tokens  final word tokens
    '''
     # normalize, lemmatize and clean
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    Create pipelines for bulding and tuning model 

    input
    None

    Returns
    gridcv  model building and tuning pipeline
    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])
    parameters = {
            'clf__estimator__n_estimators': [10, 20],
            'clf__estimator__min_samples_split': [2, 4]}

    grid_cv = GridSearchCV(pipeline, param_grid=parameters)

    return grid_cv


def evaluate_model(model, X_test, Y_test):
    '''
    Evaluates the built model displays result of quality metrics

    input
    model   model to be evaluated
    X_test  predictor variable for evaluating model
    Y_test  target variables or categories for evaluating model

    Returns
    None
    '''
    # predict to test using best params
    y_pred = model.predict(X_test)
    for i, column in enumerate(Y_test):
        print(column)
        print('\n')
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))
        
        pass
        

def save_model(model, model_filepath):
    '''
    Saves the final model as pickle file

    input
    model   model to be saved
    model_filepath  directory for saving model

    Returns
    None
    '''
    # save model
    pickle.dump(model, open(model_filepath, "wb"))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...might take a while .....')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()