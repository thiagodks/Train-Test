from scipy.sparse import csr_matrix
from collections import OrderedDict
import matplotlib.pyplot as plt
from termcolor import colored
from tqdm import tqdm
import pandas as pd
import numpy as np
import operator
import pickle
import time
import csv
import os, psutil

pid = os.getpid()
ps = psutil.Process(pid)

# base = 'Ciao-DVD'
# directory = "../../../../../media/thiagodks/HD/IC/data/CiaoDVD/ratings.data"
# sep = '::'
# header = False
# timestamp_exists = True
# test_percent = 0.2
# min_num_rating_test = 120

# base = 'Amazon-Kindle-Store'
# directory = "../../../../../media/thiagodks/HD/IC/data/Kindle Store/ratings.csv"
# sep = ','
# header = True
# timestamp_exists = True
# test_percent = 0.2
# min_num_rating_test = 120

# base = 'ML-10M'
# directory = "../../../../../media/thiagodks/HD/IC/data/ml-10m/ratings.dat"
# sep = '::'
# header = False
# timestamp_exists = True
# test_percent = 0.2
# min_num_rating_test = 120

# base = 'Netflix'
# directory = "../data/netflix/all_ratings.csv"
# sep = ','
# header = True
# timestamp_exists = True
# test_percent = 0.2
# min_num_rating_test = 120

# base = 'ML-1M'
# directory = "../../../../../media/thiagodks/HD/IC/data/ml-1m/ratings.dat"
# sep = '::'
# header = False
# timestamp_exists = True
# test_percent = 0.2
# min_num_rating_test = 120

# base = 'ML-100K'
# directory = "../../../../../media/thiagodks/HD/IC/data/ml-100k/u.data"
# sep = '\t'
# header = False
# timestamp_exists = True
# test_percent = 0.2
# min_num_rating_test = 120

base = 'Good-Books'
directory = "../../../../../media/thiagodks/HD/IC/data/GoodBooks/ratings.csv"
sep = ','
header = False
timestamp_exists = False
test_percent = 0.2
min_num_rating_test = 120

print("\nbase: ", colored(base, "green"))
print("directory: ", colored(directory, "green"))
print("sep: ", colored(sep, "green"))
print("header: ", colored(str(header), "green"))
print("timestamp_exists: ", colored(str(timestamp_exists), "green"))
print("test_percent: ", colored(str(test_percent), "green"))
print("min_num_rating_test: ", colored(str(min_num_rating_test)+"\n", "green"))

input("press enter to continue")

# Leitura do arquivo
def get_max_uid_iid(directory):

	max_uid, max_iid = -1, -1
	file = open(directory, "r")
	if header: file.readline()
	for row in file:
		values = row.split(sep)
		uid, iid = int(values[0]),int(values[1])
		if uid > max_uid: max_uid = uid
		if iid > max_iid: max_iid = iid
	return max_uid, max_iid

def save_real_id(backup_uid, backup_iid):
	with open("Train-Test_"+base+'/real-uid_'+base, 'wb') as fp:
		pickle.dump(backup_uid, fp)
	with open("Train-Test_"+base+'/real-iid_'+base, 'wb') as fp:
		pickle.dump(backup_iid, fp)

def read_archive(directory, max_uid, max_iid):

	file = open(directory, "r")
	
	if max_uid < 65535: backup_uid = np.full(max_uid+1, -1, dtype=np.int16)
	else: backup_uid = np.full(max_uid+1, -1, dtype=np.int32)
	if max_iid < 65535: backup_iid = np.full(max_iid+1, -1, dtype=np.int16)
	else: backup_iid = np.full(max_iid+1, -1, dtype=np.int32)

	uid_seq = 0
	iid_seq = 0
	cont = 0

	if header: file.readline()
	for row in tqdm(file, position=0, leave=True):
		
		values = row.split(sep)
		uid, iid, rating = int(values[0]),int(values[1]), np.float16(values[2])
		
		if backup_uid[uid] == -1:
			backup_uid[uid] = uid_seq
			uid_seq += 1

		if backup_iid[iid] == -1:
			backup_iid[iid] = iid_seq
			iid_seq += 1

		usersId[cont] = backup_uid[uid]
		itemsId[cont] = backup_iid[iid]

		ratings[cont] = rating
		if timestamp_exists: timestamp[cont] = np.uint64(values[3].replace('\n', ''))
		else: timestamp[cont] = cont
		cont += 1
		
	file.close()
	save_real_id(backup_uid, backup_iid)
	del backup_uid
	del backup_iid
	return usersId, itemsId, ratings, timestamp

try:

	numRatings = sum(1 for line in open(directory))
	max_uid, max_iid = get_max_uid_iid(directory)

	if max_uid < 65535: usersId = np.zeros(numRatings, dtype=np.uint16)
	else: usersId = np.zeros(numRatings, dtype=np.uint32)
	if max_iid < 65535: itemsId = np.zeros(numRatings, dtype=np.uint16)
	else: itemsId = np.zeros(numRatings, dtype=np.uint32)

	timestamp = np.zeros(numRatings, dtype=np.uint64)
	ratings = np.zeros(numRatings, dtype=np.float16)

	print("Types: \n\tuid: ", type(usersId[0]), "\n\tiid: ", type(itemsId[0]), "\n\trating: ", type(ratings[0]), "\n\ttimestamp: ", type(timestamp[0]))

	init_time = time.time()
	if not os.path.exists("Train-Test_"+base): os.mkdir("Train-Test_"+base)

	print("\nIniciando Leitura do arquivo... MAX("+str(numRatings)+")")
	usersId, itemId, ratings, timestamp = read_archive(directory, max_uid, max_iid)
	num_movies = len(np.unique(itemsId))
	num_users = len(np.unique(usersId))
	print("\nNúmero total de usuários: ", colored(str(num_users), 'green'))
	print("Número total de filmes: ", colored(str(num_movies), 'green'))

	ratings_matrix = csr_matrix((ratings, (usersId, itemsId)), shape=(num_users, num_movies))
	timestamp_matrix = csr_matrix((timestamp, (usersId, itemsId)), shape=(num_users, num_movies))
	print("\nratings matrix: ", ratings_matrix.shape, colored('OK', 'green'))
	print("timestamp matrix: ", timestamp_matrix.shape, colored('OK', 'green'))

	# Ordenando os usuários pelo timestamp
	users_dataset = np.unique(usersId)
	user_access = {}
	print("\nOrdenando usuários pelo timestamp...")
	for u in tqdm(users_dataset, position=0, leave=True):
		user_access[u] = np.min(timestamp_matrix[u,:].data)
		
	dict_sorted = OrderedDict(sorted(user_access.items(), key=operator.itemgetter(1), reverse = True))
	users_order = np.uint32(list(dict_sorted.keys()))

	num_users_test = int(num_users * test_percent)
	num_users_train = num_users - num_users_test

	users_train = {}
	users_test = {}

	count_test = 0
	count_train = 0
	ok = np.uint8(1)

	for i in users_order:
		if len(ratings_matrix[i].data) >= min_num_rating_test and count_test < num_users_test: 
			users_test[i] = ok
			count_test += 1
		else: 
			users_train[i] = ok
			count_train += 1

	# del dict_sorted, users_order
	print("\nNúmero de usuários do treino (esperado="+str(num_users_train)+"): ", colored(count_train, 'green'))
	print("Número de usuários do teste (esperado="+str(num_users_test)+"): ", colored(count_test, 'green'))
	if count_test < num_users_test: print(colored("[Warning]: Não há "+str(num_users_test)+" com mais de "+str(min_num_rating_test), 'yellow'))
	# Separando o treino e teste
	pos_train = []
	pos_test = []

	print("\nSeparando o treino e teste...")
	for itr in tqdm(range(0,numRatings), position=0, leave=True):
		if (usersId[itr] in users_train):
			pos_train.append(np.uint32(itr))
		elif (usersId[itr] in users_test):
			pos_test.append(np.uint32(itr))
		else:
			print(colored("Error: User ["+str(usersId[itr])+"] does not identified.", 'red'))

	# del users_train, users_test
	print("\nNúmero de ratings do treino: ", colored(len(pos_train), 'green'))
	print("Número de ratings do teste: ", colored(len(pos_test), 'green'))
	print("Número total de ratings: ", colored(numRatings, 'green'))

	# Spliting the sets
	matrix_train = np.zeros((len(pos_train),4))
	matrix_test = np.zeros((len(pos_test),4))

	# # train set
	matrix_train[:,0] = usersId[pos_train[:]]
	matrix_train[:,1] = itemsId[pos_train[:]]
	matrix_train[:,2] = ratings[pos_train[:]]
	matrix_train[:,3] = timestamp[pos_train[:]]

	# test set
	matrix_test[:,0] = usersId[pos_test[:]]
	matrix_test[:,1] = itemsId[pos_test[:]]
	matrix_test[:,2] = ratings[pos_test[:]]
	matrix_test[:,3] = timestamp[pos_test[:]]

	print("\nMatriz de ratings (treino, teste): ",matrix_train.shape, matrix_test.shape)

	# Salvando o treino e teste
	print("\nSalvando treino e teste...")
	np.savetxt("Train-Test_"+base+"/trainSet_"+base+".data", 
			   matrix_train, delimiter="::", header = str("userId::itemId::rating::timestamp"), fmt = '%.1f')
	np.savetxt("Train-Test_"+base+"/testSet_"+base+".data", 
			   matrix_test, delimiter="::", header = str("userId::itemId::rating::timestamp"), fmt = '%.1f')

	memoryUse = ps.memory_info()
	print("\nOs resultados foram salvos em: Train-Test_"+base+"/")
	print("Tempo gasto: %.2f" % (time.time() - init_time), "seg.")
	print("Memoria utilizada: %.2f" % (memoryUse[0] * 0.000001), "mb\n")

except Exception as e:
	print(colored("\n[Exception]: "+str(e)+"\n", 'red'))