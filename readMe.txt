Content:
	1)How to run the code?
	2)How to Generate Required files?


Data sets used: 

Dataset-1: (Car Evaluation)
Link: https://archive.ics.uci.edu/ml/datasets/car+evaluation


Dataset-2: (Heart Disease)
link: https://archive.ics.uci.edu/ml/datasets/heart+disease


1)How to run the code?

----------------------------------------------

Random Forest with Improved KNN:
Script: python3 hybridKNNimp.py

	For Dataset 1:
	Update Line263: currentFile = dataset1

	For Dataset 2: (default)
	Update Line263: currentFile = dataset2

----------------------------------------------

Random Forest without ImprovedKNN:
Script: python3 regularRF.py

	For Dataset 1:
	Update Line204: dataset = load_csv(dataset1)

	For Dataset 2: (default)
	Update Line204: dataset = load_csv(dataset2)

----------------------------------------------

Expected Output:
Accuracy for Different number of Tree.


2)How to Generate Required files?

	For Dataset 1:
	Run TranformationCodeForDataSet1.ipynb to generate new_hybrid.csv	

	For Dataset 2: (default)
	Run TransformationCodeForDataSet2.ipynb to generate new_processed.cleveland.data

------------------------------------------------

Important:
All files should be in same folder!

Required Files:  new_processed.cleveland.data
		new_hybrid.csv

		hybridKNNImp.py
		regularRF.py
