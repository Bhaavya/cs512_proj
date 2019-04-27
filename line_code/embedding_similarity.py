import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
import json 
from Utils import *
import os

#doc embedding is average of all entity embeddings
def get_doc_em(entity_embeddings,doc_ens,mesh_entity_path):
	doc_embeddings = np.zeros((len(doc_ens),entity_embeddings[list(entity_embeddings.keys())[0]].shape[0]))
	mesh_entity_map = None
	if mesh_entity_path is not None:
		mesh_entity_map = json.load(open(mesh_entity_path,'r'))

	ids = []
	for idx,(d,ens) in enumerate(doc_ens.items()):
		ids.append(d)
		
		for en in ens:
			try:
				doc_embeddings[idx] +=entity_embeddings[en]
			except KeyError as e:
				try:
					#mesh "entities" are mapped to mesh headings, we have embeddings for mesh headings
					if mesh_entity_map is not None:
						t = np.zeros((entity_embeddings[list(entity_embeddings.keys())[0]].shape[0],))
						for me in mesh_entity_map[en]:
							t += entity_embeddings[me]
						t /= max(len(mesh_entity_map[en]),1)
						doc_embeddings[idx] += t 

					else:
						print ('{} {} not found'.format(en,e))
						

				except KeyError as e:
					print ('{} {} not found'.format(en,e))
		if len(ens)>0:
			doc_embeddings[idx] = doc_embeddings[idx]/len(ens)
	print (doc_embeddings.shape)
	return doc_embeddings,np.array(ids)

def get_doc_entities(doc_el_paths,ont_type,pmid_path=None):
	pmids = None
	if pmid_path:
		with open(pmid_path,'r') as f:
			pmids = [pmid.strip() for pmid in f.readlines()]
	doc_ens = {}

	for doc_el_path in doc_el_paths:
		with open(doc_el_path,'r') as f:
			docs = f.readlines()
		for doc in docs:
			doc = doc.strip()
			if ont_type == 'mesh':
				sp_char1 = '\t'
				sp_char2 = ';'
			else:
				sp_char1 = ':'
				sp_char2 = ' '
			pmid = doc.split(sp_char1)[0]

			if len(doc.split(sp_char1))>1:
				if pmids is not None and str(pmid) in pmids:
					if pmid in doc_ens:
						prev_ens = doc_ens[pmid]
					else:
						prev_ens = []
					new_ens = doc.split(sp_char1)[1].split(sp_char2)
					all_ens = prev_ens.copy()
					for new_en in new_ens:
						
						if new_en not in prev_ens:
							all_ens.append(new_en)
					doc_ens[pmid] = all_ens
					# print (all_ens,doc_ens[pmid])
			else:
				if pmid not in doc_ens:
					doc_ens[pmid] = []
			
	return doc_ens 

def get_query_entities(query_el_path):
	with open(query_el_path,'r') as f:
		queries = f.readlines()
	query_ens = {}
	for i,q in enumerate(queries):
		query_ens[i] = []
		for e in q.split(', '):
			query_ens[i].append(e.strip())

	return query_ens 

def load_embeddings(embeddings_path):
	with open(embeddings_path,'r') as f:
		entity_embeddings = f.readlines()
	entity_embeddings_dict = {}
	for em in entity_embeddings[1:]:
		em = em.rstrip(' \n')
		cols = em.split(' ')
		entity_embeddings_dict[cols[0]] = np.array([float(c) for c in cols[1:]])
	return entity_embeddings_dict

def get_rel(qid,true_pos_dir):
	rel_jud = {}
	with open(os.path.join(true_pos_dir,'{}.txt'.format(qid+1)),'r') as f:
		tps = [tp.strip('\n') for tp in f.readlines()]
	for tp in tps:
		rel_jud[tp] = 1
	return rel_jud

def write_pred(sorted_res,qid,pred_dir):
	with open(os.path.join(pred_dir,'{}.txt'.format(qid+1)),'w') as f:
		for tup in sorted_res:
			f.write('{}\t{}\n'.format(tup[0],tup[1]))

def main(embeddings_path,doc_el_paths,query_el_path,true_pos_dir,pred_dir,out_path,pmid_path=None,mesh_entity_path=None,ont_type='disgenet'):
	entity_embeddings = load_embeddings(embeddings_path)
	doc_ens = get_doc_entities(doc_el_paths,ont_type,pmid_path)
	query_ens = get_query_entities(query_el_path)
	doc_embeddings,pmids = get_doc_em(entity_embeddings,doc_ens,mesh_entity_path)
	query_embeddings,ids = get_doc_em(entity_embeddings,query_ens,mesh_entity_path)
	sim_mat = cosine_similarity(query_embeddings,doc_embeddings)
	sorted_idx = np.argsort(sim_mat)[:,::-1] 
	ndcg10s = []
	for i,idx in enumerate(sorted_idx):
		sorted_res = []
		for id_ in idx:
			sorted_res.append((pmids[id_],sim_mat[i,id_]))
		rel_jud = get_rel(i,true_pos_dir)
		res = evaluate_res(sorted_res,rel_jud)
		print (i,res)
		ndcg10s.append(res['ndcg10'])
		write_pred(sorted_res,i,pred_dir)
	with open(res_path,'w') as f:
		for n in ndcg10s:
			f.write(str(n))
			f.write('\n')

if __name__ == '__main__':
	embeddings_path = '/Users/bhavya/Documents/cs512/project/data/line_embeddings/mesh_undirected_embedding_concat.csv'
	doc_el_paths = ['/Users/bhavya/Documents/cs512/project/data/jinfeng_data/data/jinfeng/doc_pubtator_entities.txt','/Users/bhavya/Documents/cs512/project/data/jinfeng_data/data/jinfeng/doc_pubmed_mh.txt']
	query_el_path = '/Users/bhavya/Documents/cs512/project/data/jinfeng_data/data/jinfeng/queries_mesh_manual_first50.txt'
	pmid_path = '/Users/bhavya/Documents/cs512/project/data/jinfeng_data/data/jinfeng/corpus_pmid.txt'
	mesh_entity_path = '/Users/bhavya/Documents/cs512/project/data/entity_mh_links.json' #for mapping mesh "entities" to mesh headings
	true_pos_dir ='/Users/bhavya/Documents/cs512/project/data/jinfeng_data/data/jinfeng/true_positives'
	pred_dir ='/Users/bhavya/Documents/cs512/project/data/results/line_predictions_mesh' 
	res_path ='/Users/bhavya/Documents/cs512/project/data/results/mesh_line_ndcg10.txt' 
	ont_type = 'mesh'

	# embeddings_path = '/Users/bhavya/Documents/cs512/project/data/line_embeddings/disgenet_undirected_embedding_concat.csv'
	# doc_el_paths = ['/Users/bhavya/Documents/cs512/project/data/disgenet_id_link.txt']
	# query_el_path = '/Users/bhavya/Documents/cs512/project/data/disgenet_query_link.txt'
	# pmid_path = '/Users/bhavya/Documents/cs512/project/data/jinfeng_data/data/jinfeng/corpus_pmid.txt'
	# mesh_entity_path = None
	# true_pos_dir ='/Users/bhavya/Documents/cs512/project/data/jinfeng_data/data/jinfeng/true_positives'
	# pred_dir ='/Users/bhavya/Documents/cs512/project/data/results/line_predictions_disgenet' 
	# res_path ='/Users/bhavya/Documents/cs512/project/data/results/disgenet_line_ndcg10.txt'
	# ont_type = 'disgenet' 

	main(embeddings_path,doc_el_paths,query_el_path,true_pos_dir,pred_dir,res_path,pmid_path,mesh_entity_path,ont_type)