import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
import pickle
nltk.download('averaged_perceptron_tagger')
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator,TransformerMixin

def load_data(database_filepath):
    '''
    Input:
        database_filepath: path+name of database file
    Output:
        X,Y:split of df by dependent variable and explanatory variables
        category_names: names of categories (list)
    '''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_response_data', engine)  
    X = df.message 
    Y = df.drop(columns=["id", "message","original", "genre"])
    category_names = Y.columns
    return X,Y,category_names

def tokenize(text):
    '''
    Input:
        text: unprocessed text entries
    Output:
        clean_tokens: tokenized and lemmatized text w/out URL and numbers
    '''   
    
    # remove url from text
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # remove numbers from the text (also incl. expression that start with a number)
    number_regex = '^[0-9]*$'
    detected_numbers = re.findall(number_regex, text)
    for number in detected_numbers:
        text = text.replace(number, "number")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Input:
        none
    Output:
        cv: grid search model based on RF classifier
    ''' 
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    parameters = {
        'clf__estimator__n_estimators': [50, 200],
        'clf__estimator__min_samples_split': [2, 4]}

    cv = GridSearchCV(pipeline, param_grid=parameters, scoring="f1_micro")
    
    return cv

    
def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Input:
        model: grid search model
        X_test: test data set
        Y_test: categorical variables for test data set
        category_names: list of names with categories
    Output:
        none
    ''' 
    y_pred = model.predict(X_test)
    for i,col in enumerate(Y_test.columns):
        print("Category:", col)
        print(classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))


def save_model(model, model_filepath):
    '''
    Input:
        model: model to be saved
        model_filepath: path to folder where model is saved
    Output:
        none
    '''    
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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