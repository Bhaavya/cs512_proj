import pandas as pd
def find_degree(edges):
	deg = {}
	for _,edge in edges.iterrows():
		try:
			deg[str(edge[0])] +=1
		except:
			deg[str(edge[0])] = 1
	return deg

def main(path,path2):
	edges = pd.read_csv(path,sep=' ',header=None)
	edges = edges.drop_duplicates()
	print (len(edges.drop_duplicates()),len(edges))
	deg = find_degree(edges)
	with open(path2,'w') as f:
		for n,d in deg.items():
			f.write('{},{}\n'.format(n,d))

	# with open(path,'w') as f:
	# 	for _,edge in edges.iterrows():
	# 		f.write('{} {} {}\n'.format(edge[0],edge[1],edge[2]))
	# 		f.write('{} {} {}\n'.format(edge[1],edge[0],edge[2]))
if __name__ == '__main__':
	main('/Users/bhavya/Documents/cs512/project/data/disgenet_parent_child.csv','/Users/bhavya/Documents/cs512/project/data/disgenet_degree.csv')




