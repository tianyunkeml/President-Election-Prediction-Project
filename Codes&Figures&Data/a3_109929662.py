import sys
import numpy as np
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
import math
import pandas.core.frame as pdf
import sklearn
from sklearn import preprocessing as ppc
from sklearn.decomposition import PCA

# following parameters were set after many trials
NUMDA_MULT = 0.08	# adjust pace of numda increase
L2_NUMDA_MULT = 2000
PCA_startN = 5    # pca number of components test start from what
TEST_RUNS = 30    # test for how many parameters to decide best n component for PCA or best penalty for regularized regression
numBin = 20
write_file = open('./output.txt','w')

# n-fold cross validation method for question (3)
def n_fold(pred_mtr,outcome,n):
	pred_mtr = ppc.scale(pred_mtr)
	outcome = ppc.scale(outcome)
	total = pred_mtr.shape[0]
	step = int(total / 
	MSE = 0
	for i in range(n):
		if i == n-1:
			test_range = list(range(step * (n - 1),step * n))
			train_range = list(range(0,step * (n - 1)))
		else:
			test_range = list(range(i * step,i * step + step))
			train_range = list(range(0,i * step)) + list(range(i * step + step,step * n))
		train_set = pred_mtr[train_range]
		test_set = pred_mtr[test_range]
		train_y = outcome[train_range]
		test_y = outcome[test_range]
		beta = np.linalg.lstsq(train_set,train_y)[0]
		y_pred = np.dot(beta,test_set.T)
		dif = test_y - y_pred
		MSE += (sum(dif * dif)) / len(dif)
	MSE = MSE / n
	return MSE

# n-fold cross validation method for question (5)
def n_fold_with_dev(pred_mtr,outcome,n,mthd):
	# mthd = 'PCA'|'L1'|'L2' for different methods
	pred_mtr = ppc.scale(pred_mtr)
	outcome = ppc.scale(outcome)
	total = pred_mtr.shape[0]
	step = int(total / n)
	if mthd == 'PCA':
		mypca = PCA()
		pca_mtr = mypca.fit(pred_mtr).components_
		pca_pred = np.dot(pred_mtr,pca_mtr.T)
	count = 0
	min_MSE = 10000
	min_testMSE = 10000
	best_count = 10000
	best_n = 10000
	best_numda = 10
	while count < TEST_RUNS:
		MSE = 0
		testMSE = 0
		if mthd != 'PCA' and count == 0:
			MSE = 3000
			testMSE = 3000
		for i in range(n):
			if i == n - 2:
				dev_range = list(range(step * (n - 2),step * (n - 1)))
				test_range = list(range(step * (n - 1),step * n))
				train_range = list(range(0,step * (n - 2)))
			elif i == n - 1:
				dev_range = list(range(step * (n - 1),step * n))
				test_range = list(range(0,step))
				train_range = list(range(step,step * (n - 1)))
			else:
				dev_range = list(range(step * i,step * i + step))
				test_range = list(range(step * (i + 1),step * (i + 2)))
				train_range = list(range(0,step * i)) + list(range(step * (i + 2),step * n))
			train_set = pred_mtr[train_range]
			test_set = pred_mtr[test_range]
			dev_set = pred_mtr[dev_range]
			train_y = outcome[train_range]
			test_y = outcome[test_range]
			dev_y = outcome[dev_range]
	
			if mthd == 'PCA':
				pca_npred = pca_pred.T[range(0,PCA_startN + count)].T
				train_set_pca = pca_npred[train_range]
				test_set_pca = pca_npred[test_range]
				dev_set_pca = pca_npred[dev_range]
				beta = np.linalg.lstsq(train_set_pca,train_y)[0]
				y_pred = np.dot(beta,dev_set_pca.T)
				y_test_pred = np.dot(beta,test_set_pca.T)
				dif = dev_y - y_pred
				dif_test = test_y - y_test_pred
				MSE += (sum(dif * dif)) / len(dif)
				testMSE += (sum(dif_test * dif_test)) / len(dif_test)
			else:
				if mthd == 'L2' and count > 0:
					clf = sklearn.linear_model.Ridge(alpha = L2_NUMDA_MULT * NUMDA_MULT * float(count) / TEST_RUNS,fit_intercept = False,normalize = False)
					L2_model = clf.fit(train_set,train_y)
					y_pred = L2_model.predict(dev_set)
					y_test_pred = L2_model.predict(test_set)
					dif = dev_y - y_pred
					dif_test = test_y - y_test_pred
					MSE += (sum(dif * dif)) / len(dif)
					testMSE += (sum(dif_test * dif_test)) / len(dif_test)
				if mthd == 'L1' and count > 0:
					clf = sklearn.linear_model.Lasso(alpha = NUMDA_MULT * float(count) / TEST_RUNS,fit_intercept = False,normalize = False)
					L1_model = clf.fit(train_set,train_y)
					y_pred = L1_model.predict(dev_set)
					y_test_pred = L1_model.predict(test_set)
					dif = dev_y - y_pred
					dif_test = test_y - y_test_pred
					MSE += (sum(dif * dif)) / len(dif)
					testMSE += (sum(dif_test * dif_test)) / len(dif_test)
		testMSE = testMSE / n
		MSE = MSE / n
		if MSE < min_MSE:
			min_MSE = MSE
			best_count = count
			min_testMSE = testMSE
		count += 1
	
	return min_testMSE
	

# ***********************QUESTION 1**********************
myFile = pd.read_csv(sys.argv[1],low_memory = False)
#To filter out state total rows
myFilter = myFile.COUNTYCODE != 0
myFile = myFile[myFilter]
col = myFile.columns
ind = myFile.index

#filter cointies with at least 30,000 people
filter1 = []
for i in ind:
	tmp = myFile.loc[i,'2011 population estimate Value']
	if isinstance(tmp,str):
		tmp = int(tmp.replace(',',''))
	if tmp > 30000:
		filter1.append(i)
myFile = myFile.loc[filter1,:]

#filter columns ending in 'Value'
filter2 = []
for i in col:
	if 'Value' in i:
		filter2.append(i)
filter2 = ['COUNTYCODE'] + filter2
myFile = myFile[filter2]

#filter rows with no NaN
filter3 = []
col = myFile.columns
ind = myFile.index
for i in ind:
	sign = 1
	for j in col:
		if j != 'COUNTYCODE':
			tmp = myFile.loc[i,j]
			if not isinstance(tmp,(str,int,float)):
				sign = 0
			else:
				if isinstance(tmp,float):
					if math.isnan(tmp) == True:
						sign = 0
				if isinstance(tmp,str):
					myFile.loc[i,j] = float(tmp.replace(',',''))			
	if sign == 1:
		filter3.append(i)
myFile = myFile.loc[filter3,:]
ind = myFile.index

write_file.write('1. TOTAL NUMBER OF COUNTIES: %d\n\n'%len(ind))

#**************************QUESTION 2***************************
count = 0
new_col = np.random.randn(len(ind),1)
log_list = []
for i in ind:
	tmp = myFile.loc[i,'Premature age-adjusted mortality Value']
	tmp = math.log(tmp)
	new_col[count,0] = tmp
	log_list.append(tmp)
	count += 1

new_df = pdf.DataFrame(new_col,columns = ['log_paamv'],index = ind)
myFile = pdf.DataFrame.merge(myFile,new_df,left_index = True,right_index = True)
log_list = pd.Series(log_list)
log_hist = log_list.hist(bins = numBin)
log_hist.plot()
plt.title('2. log_paamv HISTOGRAM: 2histogram.png')
plt.savefig('2. log_paamv HISTOGRAM: 2histogram.png')
plt.clf()
write_file.write('2. Produced a histogram to a png\n\n')
col = myFile.columns

#*************************QUETION 3***************************
rm_set = ['COUNTYCODE','log_paamv','Premature age-adjusted mortality Value', 'Premature death Value', 'Uninsured adults Value', 'Teen births Value', 'Food insecurity Value', 'Physical inactivity Value', 'Adult smoking Value', 'Injury deaths Value', 'Motor vehicle crash deaths Value', 'Drug poisoning deaths Value', 'Child mortality Value', 'Uninsured Value']
#shuffle the index
myFile = myFile.reindex(np.random.permutation(myFile.index))
log_list = myFile.log_paamv.tolist()
for item in rm_set:
	col = col.drop(item)
myFile = myFile[col]
pred_mtr = myFile.as_matrix()
MSE = n_fold(pred_mtr,log_list,10)
write_file.write('3. Non-regularized Linear Regression MSE: %.4f\n\n'%MSE)


#****************************QUESTION 4**************************
mypca = PCA(n_components = 3)
mypca.fit(ppc.scale(pred_mtr))
write_file.write('4. Percentage variance explained of first three components: ' + str(mypca.explained_variance_ratio_) + '\n\n')


#****************************QUESTION 5**************************
MSE_PCA = n_fold_with_dev(pred_mtr,log_list,10,mthd = 'PCA')
MSE_L2 = n_fold_with_dev(pred_mtr,log_list,10,mthd = 'L2')
MSE_L1 = n_fold_with_dev(pred_mtr,log_list,10,mthd = 'L1')
write_file.write('5. a) principal components regression mse: %.4f\n'%MSE_PCA)
write_file.write('5. b) L2 regularized mse: %.4f\n'%MSE_L2)
write_file.write('5. c) L1 regularized mse: %.4f\n'%MSE_L1)

write_file.close()


