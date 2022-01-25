# Product Category Classification

The product category is determined based on the description or the text input. The text is sent to the models for prediction and the output is a product category. The models are trained on description, any additional text and manufacturer. Four different models(Decision tree, KNN, SVC and Random forest) are used which will enable us to compare the results. 	

Setup and Installation: 

Download the code zip and pip install requirements.txt

Training:

All the trained models are enclosed in the folder and can be loaded for the predictions.

If the models need to be trained on a different data then run the file, Train_SaveModels.py and change the file path and other required fields in the code

Testing the models:

Using FastAPI:
	
  Open a command prompt in the folder with main.py file and execute the following command.
  
		command: -uvicorn main:app
	
  A temporary http link will be given where the solution can be tested. Take the link and test the solution in ‘Postman’ as ‘Post’ service with the raw input in Json format as shown below.

	Input: 
    {
      "test_string" : "<test string>",
      "modelName" : "DecisionTree/SVC/KNN/RandomForest"
    }
    
    
Without FastAPI:
	
  Run the main function in “classifier.py” with test_string and modelName as input parameters as shown below.
	Category = Classifier.main("<test string>","DecisionTree/SVC/KNN/RandomForest")


Output Product categories:

-Bicycles

-Washing Machines

-Contact Lenses

-USB memory
