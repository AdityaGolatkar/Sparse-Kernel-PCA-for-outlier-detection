# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 01:41:20 2017

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
			elastic = ElasticNet(alpha=1, l1_ratio=r, max_iter=20000)
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
	zero = np.load("f_0.npy")
	one = np.load("f_1.npy")
	two = np.load("f_2.npy")
	three = np.load("f_3.npy")
	four = np.load("f_4.npy")
	five = np.load("f_5.npy")
	six = np.load("f_6.npy")
	seven = np.load("f_7.npy")
	eight = np.load("f_8.npy")
	nine = np.load("f_9.npy")

	#for 1 
	#gamma = 1/(2*3348199.75142)
	#for 9
	gamma = 1/(2*5414354.61271)
    
	points_per_digit = 400
	#for 1,6
	no_of_ev_of_gram = 15
	#for 0
	no_of_ev_of_gram = 15
	threshold_1 = 0.29
	threshold_1s = 0.28
	#for 0.5
	#threshold_1n = 0.5
	#for 0.6
	#threshold_1n = 0.5
	#for 0.7
	threshold_1n = 0.5
	threshold_9 = 0.26
	threshold_9s = 0.26
	#for 0.5
	#threshold_9n = 0.52
	#for 0.6
	threshold_9n = 0.52
    
	threshold = threshold_9n
	
	data = nine[0:points_per_digit,0:]
    
	testpoints_per_samedigit = 450
	test_data_same = nine[points_per_digit:points_per_digit+testpoints_per_samedigit,0:]
    
	testpoints_per_diffdigit = 50
	diffdata1 = np.append(three[0:testpoints_per_diffdigit,0:],two[0:testpoints_per_diffdigit,0:],axis=0)
	diffdata2 = np.append(zero[0:testpoints_per_diffdigit,0:],four[0:testpoints_per_diffdigit,0:],axis=0)
	diffdata3 = np.append(five[0:testpoints_per_diffdigit,0:],six[0:testpoints_per_diffdigit,0:],axis=0)  
	diffdata4 = np.append(seven[0:testpoints_per_diffdigit,0:],eight[0:testpoints_per_diffdigit,0:],axis=0)    
	test_data_diff1 = np.append(diffdata1,diffdata2,axis=0)
	test_data_diff2 = np.append(diffdata3,diffdata4,axis=0)
	test_data_diff = np.append(test_data_diff1,test_data_diff2,axis=0)
	test_data_diff = np.append(test_data_diff,one[0:testpoints_per_diffdigit,0:],axis=0)
    
	all_data = np.append(data,test_data_same,axis = 0)
	all_data = np.append(all_data,test_data_diff,axis = 0)

	print(all_data.std(axis=0))    
	#data = (data - all_data.mean(axis=0)) / all_data.std(axis=0) 
	#test_data_same = (test_data_same - all_data.mean(axis=0)) / all_data.std(axis=0)
	#test_data_diff = (test_data_diff - all_data.mean(axis=0)) / all_data.std(axis=0)

	G_not_cen, G_cen = gram_generate(data,gamma)
	print(G_cen)
	sio.savemat('G_cen.mat', {'G_cen':G_cen})    
	actual_e_val,actual_e_vec = gram_eigen_vectors(G_cen,no_of_ev_of_gram)
	print(actual_e_val[0:10])  
	print(actual_e_val[11:20])  
	l1_ratio = 0.6
	sparse_e_vec = naive_spca(G_cen,no_of_ev_of_gram,l1_ratio) 
	#sparse_e_vec = actual_e_vec
    
	thresh_e_vec = np.zeros(actual_e_vec.shape)
	tot_ct = 0     
	for i in range(no_of_ev_of_gram):          
		ct = 0		 
		for j in range(points_per_digit):		                 
			if abs(sparse_e_vec[j,i]) > 0.0000001:
			     ct=ct+1
		print(ct)
		tot_ct = tot_ct+ct      

	tot_ct = int(tot_ct/no_of_ev_of_gram)   
	#tot_ct = 23    
    
	for i in range(no_of_ev_of_gram):          
		abs_ev_i = np.abs(actual_e_vec[:,i])
		idx = abs_ev_i.argsort()[-tot_ct:][::-1]
		thresh_e_vec[idx,i] = actual_e_vec[idx,i]   

	alphas = thresh_e_vec
	alpha_sum = np.sum(alphas,0)
	G_row = np.sum(G_not_cen,0)/points_per_digit
	G_sum = np.sum(G_row)/points_per_digit
	alpha_G_row = np.dot(G_row,alphas)
    
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
	
	print(tot_ct)
    
main()