# Alphabet-Drawing-Recognition

Provides interface for drawing an uppercase english alphabet.
The drawing is fed to a CNN (Convolutional Neural Network) model which predicts what alphabet is drawn.


STEPS FOR EXECUTION:

	RUN THE RECOGNIZER:
		-> run 'recognizer.py' using the command: python recognizer.py
		-> the code loads a trained CNN model from the directory './saved_models'
		-> an interface will appear where you can draw an uppercase alphabet and predict it using the loaded model.

	TRAINING THE MODEL:
		-> download the data from 'https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format/download'
		-> store the downloaded csv file in the directory './downloaded_data'
		-> run 'dataset_creator.py' using the command: python dataset_creator.py
		-> this will create a directory './AlphabetDataset' which stores the images in separate directories for each alphabet
		-> after the dataset is created, we can run 'model_notebook.ipynb' to train the CNN model
		-> after a model is trained, save it in the directory './saved_models' to use it in 'recognizer.py' for prediction
	

DATASET LINK:
	https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format/download


RESULTS:	
	10 fold cross validation accuracy of the model: 95.1%
	
# SCREENSHOTS:
	![alt text](https://github.com/soumyaporel/Alphabet-Drawing-Recognition/blob/main/screenshots/Screenshot%20from%202021-08-22%2013-00-11.png?raw=true)
	![alt text](https://github.com/soumyaporel/Alphabet-Drawing-Recognition/blob/main/screenshots/Screenshot%20from%202021-08-22%2013-00-33.png?raw=true)
	![alt text](https://github.com/soumyaporel/Alphabet-Drawing-Recognition/blob/main/screenshots/Screenshot%20from%202021-08-22%2013-00-41.png?raw=true)
	
