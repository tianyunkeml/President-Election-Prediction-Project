1. What are in the folder
	county_facts.csv: demographic factors of counties
	primary_results.csv: voting results for each candidate in each county
	popularity_train.xlsx: popularity values of each candidate in each county for training
	sentiment_train.xlsx: sentiment scores of each candidate in each county for training
	candidateData.py: python module to collect data from above files
	PCA_Mapping.py: main code to train the multiple linear regression model (uses PCA mapping and Ridge regression), and finds the minimum test MSE of the model
	other files if any: currently useless

2. How to run
	To get the multiple linear regression model for each candidate, in the directory of the codes and run:
		python3 PCA_Mapping.py
	which returns the minumum test MSE for each candidate, together with the corresponding number of PCA components and alpha value of Ridge regression:Donald Trump:
		Best components: 47\Best alpha: 8.000000
		Best Test MSE:0.144568
		Ted Cruz:
		Best components: 36\Best alpha: 64.000000
		Best Test MSE:0.163163
		Hillary Clinton:
		Best components: 51\Best alpha: 0.062500
		Best Test MSE:0.136950
		Bernie Sanders:
		Best components: 46\Best alpha: 1.000000
		Best Test MSE:0.294604
	
