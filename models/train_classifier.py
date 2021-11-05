import sys
import nltk
import pickle
import time
nltk.download(['punkt', 'wordnet'])
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_data(database_filepath):
    '''
    Load data form SQLite database.

    Parameters
    ----------
    database_filepath : string
        filepath of database.

    Returns
    -------
    X : pd.DataFrame
        varisable to be studied.
    Y : pd.DataFrame
        variables to create the model.
    category_names : list
        list with categories names.

    '''
    name = 'sqlite:///' +str(database_filepath)
    engine = create_engine(name)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.drop(['index','id','message','original','genre'],axis=1)
    category_names = Y.columns
    #print(X.shape,Y.shape)
    #print(Y.columns)
    return X,Y,category_names

def tokenize(text):
    '''
    Tokenizer to clean and convert text into clean and divided data that a
    a model can understand.

    Parameters
    ----------
    text : string
        text o be tokenized.

    Returns
    -------
    clean_tokens : string
        cleand text.

    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    '''
    Defines a Pipeline for the model and runs a grid search in it.

    Returns
    -------
    cv : Pipeline
        Pipeline model with grid search.

    '''
    pipeline =  Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    #parameters = {
    #    'vect__ngram_range':((1,1)),#((1, 1), (1, 2)),
    #    'vect__max_df':[(0.75)],# (0.5, 0.75), #0.75
    #    'vect__max_features':(10000,30000), #(None, 5000, 10000), #10000
    #    'clf__estimator__learning_rate':[1.0], #[0.5,1.0], # 1.0
    #    'clf__estimator__n_estimators': [100] # [50,100]
    #}
    
    #cv = GridSearchCV(pipeline, param_grid=parameters)
    cv = pipeline
    return cv
    
    


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Fit the model and print results

    Parameters
    ----------
    model : Pipeline.fit
        
    X_test : pd.DataFrame
        messages dataframe.
    Y_test : pd.DataFrame
        categories dataframe.
    category_names : list
        list of categories names.

    Returns
    -------
    None.

    '''
    
    # predict on test data
    y_pred = pd.DataFrame(model.predict(X_test))
    # print clasification repory
    for i in range(Y_test.shape[1]):
        print('Column: %s , %s' %(i,category_names[i]))
        print(classification_report(Y_test.iloc[:,i], y_pred.iloc[:,i]))

    accuracy = (y_pred.values == Y_test.values).mean()
    print('The model accuracy score is %0.4f' %(accuracy))

def save_model(model, model_filepath):
    '''
    Fit the model and print results

    Parameters
    ----------
    model : Pipeline.fit
        Model you want to export.
	model_filepath : string
        Path containing name of the file.
	'''
    with open(model_filepath,'wb') as f:
        pickle.dump(model,f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        start = time.time()
        model.fit(X_train, Y_train)
        print('runtime in seconds: %0.2f' %(time.time()-start))
        #print('best parameters: '+ str(model.best_params_))
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