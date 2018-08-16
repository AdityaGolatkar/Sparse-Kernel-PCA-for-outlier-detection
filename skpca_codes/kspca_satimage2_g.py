# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 21:12:11 2017

@author: Rudrajit
"""

import numpy as np
from sklearn.linear_model import ElasticNet
import scipy.io as sio


#####################################
# x1 = input vector 1
# x2 = input vector 2
# gamma = parameter of the RBF kernel
# k = value of the kernel function
#####################################
def kernel(x1,x2,gamma):   
	k = np.exp(-gamma*(np.linalg.norm(x1-x2))**2)
	return k

###############################################
# X = data matrix with each point along the row
# gamma = parameter of the RBF kernel
# G = non centered Gram Matrix
# G_mean_shifted = Centred Gram Matrix
###############################################

def gram_generate(X,gamma):
	no_of_points = X.shape[0]
	print(no_of_points)
	G = np.zeros((no_of_points,no_of_points))
	for i in range(no_of_points):
		for j in range(no_of_points):
			G[i,j] =  kernel(X[i,:],X[j,:],gamma)

	sigma = 0            
	for i in range(no_of_points):
		for j in range(no_of_points):
			sigma =  sigma + (np.linalg.norm(X[i,:]-X[j,:]))**2
	print(sigma/(no_of_points*no_of_points))
            
	N = G.shape[0]
	G_row = np.sum(G,0)/N
	G_sum = np.sum(G_row)/N    
	G_mean_shifted = np.zeros((no_of_points,no_of_points)) 
	for i in range(no_of_points):
		for j in range(no_of_points):
			G_mean_shifted[i,j] = G[i,j] - G_row[i] - G_row[j] + G_sum
    
	return G, G_mean_shifted


#################################################
# G = Centred Gram Matrix
# eigen_value = Eigen Value of the Gram Matrix
# eigen_vecktor = Eigen Vector of the Gram Matrix
#################################################

def gram_eigen_vectors(G,num_eig_vecs):
	eigen_value, eigen_vector = np.linalg.eigh(G)
	idx = eigen_value.argsort()[::-1] 
	eigen_value = eigen_value[idx]   
	eigen_vector = eigen_vector[:,idx]   
	for i in range(num_eig_vecs):
			eigen_vector[:,i] = (eigen_vector[:,i]/np.sqrt(eigen_value[i]))
            
	return eigen_value[0:num_eig_vecs],eigen_vector[:,0:num_eig_vecs]

#######################################
# Data = in this case is the Gram Matrix
# no_of_ev_of_gram = is the number of 
# eigen vectors of the Gram Matrx 
# returned matrix = contain the Sparse
# Eigen Vectors in columns
#######################################

def naive_spca(data,no_of_ev_of_gram,r):

	no_of_points = data.shape[0]
	feature_size = data.shape[1]
	sparse_pc_old = np.zeros((feature_size,no_of_ev_of_gram))
	################################
	#SVD gives u s v.T and not u s v
	################################
	u,s,v = np.linalg.svd(data,full_matrices=0)
	print(s)
	s2 = np.diag(s)    
	for i in range (s2.shape[0]):
		s2[i,i] = 1/np.sqrt(s[i])
	#print(v[0:5,:].T)
	#print(np.dot(v[0:5,:],v[0:5,:].T))    
	
	j = 0
	sparse_pc_list = []
	sparse_pc_list.append(sparse_pc_old)
	
	while(j < 6):
		sparse_pc = np.zeros((feature_size,no_of_ev_of_gram))
		if j == 0:
			a = v[0:no_of_ev_of_gram,0:].T
		j=j+1
		for i in range(no_of_ev_of_gram):
			y = data.dot(a[0:,i])
			#y = np.dot(data,a[0:,i])
			elastic = ElasticNet(alpha=1, l1_ratio=r, max_iter=40000)
			elastic.fit(data*np.sqrt(2*no_of_points),y*np.sqrt(2*no_of_points))
			pc = elastic.coef_
			#print(pc)
			sparse_pc[0:,i] = pc
		u1,s1,v1=np.linalg.svd(np.dot(np.dot(np.dot(s2,u.T),np.dot(data.T,data)),sparse_pc),full_matrices=0)
		#u1,s1,v1=np.linalg.svd(np.dot(np.dot(data.T,data),sparse_pc),full_matrices=0)
		#pdb.set_trace()
		#a = u1[0:,0:sig_dim].dot(v1)
		a = np.dot(np.dot(u,s2),u1.dot(v1))
		#a = u1.dot(v1)
		#print(sparse_pc)
		if ((np.linalg.norm(sparse_pc-sparse_pc_list[j-1],ord='fro')))<0.0008:
			#print((np.linalg.norm(sparse_pc-sparse_pc_list[j-1],ord='fro')))
			sparse_pc_list.append(sparse_pc)	
			break
		sparse_pc_list.append(sparse_pc)
	#print(sparse_pc_list)
	nrm = np.sqrt(np.sum(sparse_pc_list[len(sparse_pc_list)-1]*sparse_pc_list[len(sparse_pc_list)-1],axis=0))
	#print(nrm)
	#sparse_pc_list[len(sparse_pc_list)-1]=sparse_pc_list[len(sparse_pc_list)-1]/nrm
	for i in range(no_of_ev_of_gram): 
		sc = np.sqrt(np.dot(sparse_pc_list[len(sparse_pc_list)-1][:,i].T,np.dot(data,sparse_pc_list[len(sparse_pc_list)-1][:,i])))           
		sparse_pc_list[len(sparse_pc_list)-1][:,i] = (1/sc)*sparse_pc_list[len(sparse_pc_list)-1][:,i]		
		#sparse_pc_list[len(sparse_pc_list)-1][:,i] = (1/np.sqrt(s[i]))*sparse_pc_list[len(sparse_pc_list)-1][:,i]
	return sparse_pc_list[len(sparse_pc_list)-1]


#####################################################
# alphas = contains the eigen vectors of G in columns
# test_point should be a row vector
# G_not_cen = Gram Matrix
# test_point = testing point
# gamma = parameter of the kernel
# returned value = reconstruction error
######################################################

def recon_error(test_point,data,alphas,sum_alpha,alpha_G_row,G_sum,gamma):
	n = data.shape[0]
	k = np.zeros(n)
	for j in range(n):
		k[j] = kernel(test_point,data[j,:],gamma)

	f = np.dot(k,alphas) - np.dot(sum_alpha,(np.sum(k)/n)-G_sum) - alpha_G_row

	err = kernel(test_point,test_point,gamma) - 2*(np.sum(k)/n) + G_sum - f.T.dot(f)
    
	return err    


def main():   
	points_per_digit = 400

	# sparse
	no_of_ev_of_gram = 15
	# for 0.1 
	#threshold = 0.36
	# for 0.2 
	#threshold = 0.3
	# for 0.25 
	#threshold = 0.3
	# for 0.15 
	threshold = 0.3
    
    # non-sparse    
	#no_of_ev_of_gram = 15
	#threshold = 0.355

	X = sio.loadmat('X_satimage2.mat')
	X = X['X']
	Y = sio.loadmat('Y_satimage2.mat')
	Y = Y['y']
	inliers = np.zeros([5803-71,36])
	inliers_ct = 0
	outliers = np.zeros([71,36])
	outliers_ct = 0
	for i in range(5803):
		#print(Y[i])
		if Y[i] < 0.5:
			inliers[inliers_ct,:] =  X[i,:]
			inliers_ct = inliers_ct+1
		else:
			outliers[outliers_ct,:] =  X[i,:]
			outliers_ct = outliers_ct+1
	inliers_outliers = np.append(inliers,outliers,axis=0)
	#inliers = (inliers - inliers_outliers.mean(axis=0)) / inliers_outliers.std(axis=0) 
	#inliers = inliers + np.random.uniform(-0.05,0.05,inliers.shape)
	#outliers = (outliers - inliers_outliers.mean(axis=0)) / inliers_outliers.std(axis=0)
	#outliers = outliers + np.random.uniform(-0.05,0.05,outliers.shape)
    
	data = inliers[0:points_per_digit,0:]
    
	sigma = 0            
	for i in range(points_per_digit):
		for j in range(points_per_digit):
			sigma =  sigma + (np.linalg.norm(data[i,:]-data[j,:]))**2
	gamma = 1/(2*sigma/(points_per_digit*points_per_digit))
    
	G_not_cen, G_cen = gram_generate(data,gamma)
	print(G_cen)
	sio.savemat('G_cen.mat', {'G_cen':G_cen})    
	actual_e_val,actual_e_vec = gram_eigen_vectors(G_cen,no_of_ev_of_gram)
	print(actual_e_val[0:10])  
	print(actual_e_val[11:20])  
	l1_ratio = 0.1
	#sparse_e_vec = actual_e_vec
	sparse_e_vec = naive_spca(G_cen,no_of_ev_of_gram,l1_ratio) 
	np.save('sparse_e_vec.npy',sparse_e_vec)
	#sparse_e_vec = np.load('sparse_e_vec.npy')
    
	alphas = sparse_e_vec
	alpha_sum = np.sum(alphas,0)
	G_row = np.sum(G_not_cen,0)/points_per_digit
	G_sum = np.sum(G_row)/points_per_digit
	alpha_G_row = np.dot(G_row,alphas)
     
	for i in range(no_of_ev_of_gram):          
		ct = 0		 
		for j in range(points_per_digit):		                 
			if abs(sparse_e_vec[j,i]) > 0.0000001:
			     ct=ct+1
		print(ct)
        
	for i in range(no_of_ev_of_gram):          
		ct = 0		 
		for j in range(points_per_digit):		                 
			if abs(actual_e_vec[j,i]) > 0.0000001:
			     ct=ct+1
		print(ct)

	testpoints_per_samedigit = 500 # 5803-71-400
	test_data_same = inliers[points_per_digit:points_per_digit+testpoints_per_samedigit,0:]
	#test_data_same = data 
    
	testpoints_per_diffdigit = 71
	test_data_diff = outliers[0:testpoints_per_diffdigit,0:]

    
	correct_same = 0

	for i in range(test_data_same.shape[0]):
		
		error = recon_error(test_data_same[i,0:],data,alphas,alpha_sum,alpha_G_row,G_sum,gamma)
        
		if (abs(error)) <= threshold:
			correct_same = correct_same + 1
		print(error)            
		print(i+1)
		print(correct_same)
  
	correct_diff = 0

	for i in range(test_data_diff.shape[0]):
		
		error = recon_error(test_data_diff[i,0:],data,alphas,alpha_sum,alpha_G_row,G_sum,gamma)
	
		if (abs(error)) >= threshold:
			correct_diff = correct_diff + 1
		print(error)            
		print(i+1)
		print(correct_diff)

	percent_same_correct = correct_same/test_data_same.shape[0]
	print("Same Accuracy :",percent_same_correct)    

	percent_diff_correct = correct_diff/test_data_diff.shape[0]
	print("Diff Accuracy :",percent_diff_correct)
    
	percent_correct = (correct_same+correct_diff)/(test_data_same.shape[0]+test_data_diff.shape[0])
	print("Overall Accuracy :",percent_correct)
    
	for i in range(no_of_ev_of_gram):          
		ct = 0		 
		for j in range(points_per_digit):		                 
			if abs(sparse_e_vec[j,i]) > 0.000001:
			     ct=ct+1
		print(ct)
        
	for i in range(no_of_ev_of_gram):          
		ct = 0		 
		for j in range(points_per_digit):		                 
			if abs(actual_e_vec[j,i]) > 0.000001:
			     ct=ct+1
		print(ct)    
	
main()