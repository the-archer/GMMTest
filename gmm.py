from numpy import *
import pypr.clustering.gmm as gmm
import cPickle as pickle

def load_data(path, meta):
	x=empty((1500*100, 12))
	f1 = open(meta, 'r')
	nvec=0
	for line in f1:
		s=line.rstrip('\n')
		f2=open(path+s, 'r')
		for line in f2:
			line=line.rstrip('\n')
			coeff = line.split()
			if(len(coeff)==12):
				if(nvec>=150000):
					x=append(x, coeff, axis=0)
				x[nvec]=coeff
				nvec+=1
			else:
				print "Not 12"
				print len(coeff)

		f2.close()
	f1.close()
	print nvec
	return x



path="/home/simrat/speechdata/english_digits/Training_data/cepstrals/"
meta=path+"meta_data/ceplist.txt"

x=load_data(path, meta)

center_list, cov_list, p_k, logLL =gmm.em(X=x, K=16, max_iter=3, verbose=True)


with open('gmm.pickle', 'wb') as f:
	pickle.dump([center_list, cov_list, p_k], f)


# print center_list
# print cov_list
# print p_k
# print logLL