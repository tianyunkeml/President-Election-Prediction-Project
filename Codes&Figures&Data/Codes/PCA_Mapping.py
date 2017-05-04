# Use 'sudo python3 PCA_Mapping.py' to run it. Make sure all libraries insta#lled

import numpy as np
import scipy.stats as ss
import pandas as pd
import math
import pandas.core.frame as pdf
import sklearn
from sklearn import preprocessing as ppc
from sklearn.decomposition import PCA
import candidateData

PCA_START = 1
TEST_RUNS = 1
L2_START = -15
L2_END = 23
L2_RUNS = 37
L2_STEP = (float)(L2_END - L2_START) / L2_RUNS
CANDIDATES = ['Donald Trump','Ted Cruz','Hillary Clinton','Bernie Sanders']

def n_fold_with_dev(pred_mtr,outcome,n,mthd):
	# mthd = 'PCA'|'L1'|'L2' for different methods
	pred_mtr = ppc.scale(pred_mtr)
	outcome = ppc.scale(outcome)
	outcome = np.asarray(outcome)
	total = pred_mtr.shape[0]
	step = int(total / n)
	count = 0
	min_MSE = np.inf
	min_testMSE = np.inf
	best_count = np.inf
	best_n = np.inf
	best_numda = np.inf
	best_alpha = np.inf
	while count < TEST_RUNS:
		for p in range(L2_RUNS): 
			if mthd == 'PCA':
				mypca = PCA(PCA_START + count)
				pca_mtr = mypca.fit(pred_mtr).components_
				pca_npred = np.dot(pred_mtr,pca_mtr.T)
			MSE = 0
			testMSE = 0
			if mthd != 'PCA' and count == 0:
				MSE = np.inf
				testMSE = np.inf
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
					train_set_pca = pca_npred[train_range]
					test_set_pca = pca_npred[test_range]
					dev_set_pca = pca_npred[dev_range]
					clf = sklearn.linear_model.Ridge(alpha = 2 ** (L2_START + p),fit_intercept = False,normalize = False)
					model = clf.fit(train_set_pca,train_y)
					y_pred = model.predict(dev_set_pca)
					y_test_pred = model.predict(test_set_pca)
					dif = dev_y - y_pred
					dif_test = test_y - y_test_pred
					MSE += (sum(dif * dif)) / len(dif)
					testMSE += (sum(dif_test * dif_test)) / len(dif_test)
			testMSE = testMSE / n
			MSE = MSE / n
			if MSE < min_MSE:
				min_MSE = MSE
				best_count = PCA_START + count
				min_testMSE = testMSE
				best_alpha = 2 ** (L2_START + p)
				best_model = model
			#print('Components: %d\nALPHA: %f\nMSE: %f\n'%(PCA_START + count,2 ** (L2_START + p),MSE))
		count += 1
	print('Best components: %d\Best alpha: %f'%(best_count,best_alpha))
	full_pca = PCA()
	pca_mtr = full_pca.fit(pred_mtr).components_
	pca_mtr = pca_mtr[range(0,best_count)]
	pca_mtr = pca_mtr.T	
	print('Best Test MSE:%f'%min_testMSE)

	return [min_testMSE,pca_mtr,best_model]

def get_mapping(X,Y,n = 10,with_MSE = False):
	PCA_mapping = n_fold_with_dev(X, Y, n, 'PCA')
	if with_MSE:
		return PCA_mapping
	return PCA_mapping[1]

def my_main(candList):
	outFile = open('./PCA_out.txt','w')
	PCA_Dict = {}
	for name in candList:
		print(name + ':')
		PCA_Dict[name] = []
		dataMtr = candidateData.getMatrix(name,'train')[0]
		sent = dataMtr[0]['score_x']
		pop = dataMtr[0]['score_y']
		X = dataMtr[0].as_matrix()
		Y = dataMtr[1].values.T.tolist()
		PCA_mapping = get_mapping(X,Y,n = 10,with_MSE = True)
		outFile.write(name + ':\nMSE: %f\nPCA_Matrix:\n'%PCA_mapping[0])
		outFile.write(str(PCA_mapping[1]) + '\n\n')
		PCA_Dict[name].append(PCA_mapping[1])
		PCA_Dict[name].append(PCA_mapping[2])
	outFile.close()
	return PCA_Dict

def PrimaryPredict(CANDIDATE_LIST):
	model_dict = my_main(CANDIDATES)
	Pred_Dict = {}
	for name in CANDIDATE_LIST:
		dataMtr = candidateData.getMatrix(name,'test')[1]
		X = dataMtr[0].as_matrix()
		pca_mtr = np.dot(X,model_dict[name][0])
		pred = model_dict[name][1].predict(pca_mtr)
		Pred_Dict[name] = pred
		df_pred = [dataMtr[3],dataMtr[2],pred]
		df_pred = pd.DataFrame.from_records(df_pred)
		df_pred = df_pred.T
		df_pred.to_csv('./Predict_' + name + '.csv')
	return Pred_Dict

if __name__ == '__main__':
	print(PrimaryPredict(CANDIDATES))
	
