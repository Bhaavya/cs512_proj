import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(embeddings_path):
	with open(embeddings_path,'r') as f:
		entity_embeddings = f.readlines()
	entities = []
	embeddings = []
	for i,em in enumerate(entity_embeddings[1:]):
		em = em.rstrip(' \n')
		cols = em.split(' ')
		entities.append(cols[0])
		embeddings.append(np.array([float(c) for c in cols[1:]]))
	return np.array(embeddings),entities

def main(embeddings_path):
	embeddings,entities = load_embeddings(embeddings_path)
	sim_mat=cosine_similarity(embeddings[500].reshape(1,-1),embeddings)
	# sim_mat = cosine_similarity(embeddings[3369].reshape(1,-1),embeddings)
	sim_mat[0,500] = 0
	# np.fill_diagonal(sim_mat,0)
	print (entities[500])

	for i,s in enumerate(sim_mat.argmax(axis=1)):
		# if sim_mat[i,s] > 0.5:
			# print (i,entities[1000+i])
			print (entities[s],sim_mat[i,s])
	# print('='*20)
	

if __name__ == '__main__':
	embeddings_path = '/Users/bhavya/Documents/cs512/project/data/line_embeddings/mesh_undirected_embedding_concat.csv'
	main(embeddings_path)