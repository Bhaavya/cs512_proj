import numpy as np 

def norm_embeddings(embeddings_path):
	with open(embeddings_path,'r') as f:
		entity_embeddings = f.readlines()
	embeddings = {}
	for i,em in enumerate(entity_embeddings[1:]):
		em = em.rstrip(' \n')
		cols = em.split(' ')
		em = np.array([[float(c) for c in cols[1:]]])
		embeddings[cols[0]] = em/np.linalg.norm(em)
	return embeddings

def main(o1_embeddings_path,o2_embeddings_path,concat_embeddings_path):
	o1_embeddings = norm_embeddings(o1_embeddings_path)
	o2_embeddings = norm_embeddings(o2_embeddings_path)
	print (len(o1_embeddings),len(o2_embeddings))
	concat_embeddings = {}
	tot_dim = 0
	for en,em1 in o1_embeddings.items():
		try:
			em2=o2_embeddings[en]
		except:
			print ("{} not found in o2")
		em = np.zeros((1,em1.shape[1]+em2.shape[1]))
		tot_dim = em.shape[1]
		em[0,:em1.shape[1]] = em1
		em[0,em1.shape[1]:] = em2
		concat_embeddings[en] = em 
	with open(concat_embeddings_path,'w') as f:
		f.write('{} {}\n'.format(len(concat_embeddings),tot_dim))
		for en,em in concat_embeddings.items():
			f.write('{}'.format(en))
			for val in em[0]:
				f.write(' {0:.5f}'.format(val))
			f.write('\n')	

if __name__ == '__main__':
	o1_embeddings_path = '/Users/bhavya/Documents/cs512/project/data/line_embeddings/mesh_undirected_embedding_o1.csv'
	o2_embeddings_path = '/Users/bhavya/Documents/cs512/project/data/line_embeddings/mesh_undirected_embedding_o2.csv'
	concat_embeddings_path = '/Users/bhavya/Documents/cs512/project/data/line_embeddings/mesh_undirected_embedding_concat.csv'

	main(o1_embeddings_path,o2_embeddings_path,concat_embeddings_path)