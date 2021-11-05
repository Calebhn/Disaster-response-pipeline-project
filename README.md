# Disaster-response-pipeline-project
 
This git genrates a model that clasifies messages or tweets after a disaster (in this case this data is provided by Figure Eight). to generate the model, the scripts useas NLP (Natural Language Procesing) pipline to clasifiy the messages.  after the model is created, a ".pkl" file is created to then be used to visualize and posprocess the model results in a web app using Flask. 

**The overall acurracy obtained by the model is arround 94.71% (this value will change depending on the division of the test and train data).**

## Project Subdivision

The python code contains 3 main processes:
### 1. ETL Pipeline generation
 * Extract data from a .csv file, Transform this data in a readable format for the ML pipeline and load cleand data into a SQLite database.
   - Folder: data
### 2. ML Pipeline
 * Loads the SQLite database and divides it into test and train data. after that it defines a pipeline to then apply a GridSearch to various parameters to find the highest accuracy of the model. Fits the data into a model using the pipeline and exports it into a .pkl file.
   - Folder: model
### 3. Web app to visualize results
 * Helps visualize the results of the model created before. 
   - Folder: app

## Scripts Execution

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
   
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves
    
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.

    - `python run.py`

3. Go to http://0.0.0.0:3001/ 
