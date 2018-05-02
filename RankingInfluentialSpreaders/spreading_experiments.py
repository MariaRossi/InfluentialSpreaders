import sys
import operator
import math
import os
import numpy as np
import re
import shutil
import networkx as nx
import glob
import itertools
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import subprocess
import os.path as path

class Graph:

	def __init__(self,dataset,sir_param_dict):

		self.graph=nx.Graph()
		self.graph=nx.read_weighted_edgelist(dataset)

		#get dataset name
		dataset=dataset.rstrip()
		dataset_array=dataset.split('/')
		dataset_name=dataset_array[-1]
		# remove '.txt' from the end
		self.dataset_name=dataset_name[:-4]

		self.nodes=nx.number_of_nodes(self.graph)
		self.beta=float(sir_param_dict['beta'])
		self.gamma=float(sir_param_dict['gamma'])
		self.n_iter=int(sir_param_dict['n_iter'])
		self.output_dir=str(sir_param_dict['output_dir'])
		self.kcore_length=0
		#kcore
		self.kcore_dict={}
		self.max_kcore=0
		self.top_kcore_nodes=[]

	def find_kcore(self):

		self.kcore_dict = nx.core_number(self.graph)
		self.max_kcore = max(self.kcore_dict.items(), key=operator.itemgetter(1))[1]
		kcore_subgraph = nx.k_core(self.graph)
		self.top_kcore_nodes = list(kcore_subgraph)
		self.top_kcore_nodes = [ int(x) for x in self.top_kcore_nodes ]
		self.kcore_length=len(self.top_kcore_nodes)

	def get_top_nodes(self,number_of_nodes,metric, file=None):

		metric_dict={}
		top_nodes=[]

		if (metric=='degree'):

			metric_dict=self.graph.degree()

		elif (metric=='pagerank'):

			metric_dict=nx.pagerank(self.graph)

		elif (metric=='closeness'):

			metric_dict=nx.closeness_centrality(self.graph)

		elif (metric=='eigenvector'):

			metric_dict=nx.eigenvector_centrality(self.graph)

		elif (metric=='kcore'):

			metric_dict=self.kcore_dict

		#metric input from user
		else:
			if (file is not None):
				with open(file) as f:

					for line in f:

						line=line.rstrip()
						line_array=line.split(" ")
						metric_dict[int(line_array[0])]=int(math.ceil(float(line_array[1])))
			else:
				raise ValueError('Metric dictionary file was not given...')

		max_metric= max(metric_dict.items(), key=operator.itemgetter(1))[1]
		s = [(k, metric_dict[k]) for k in sorted(metric_dict, key=metric_dict.get, reverse=True)]
		id_s=0
		for k,v in s:

			if (id_s <number_of_nodes):

				top_nodes.append(int(k))
				id_s+=1
		return top_nodes

	def run_sir(self,path_to_sir_code,node_set,method,dataset):

		os.chdir(path_to_sir_code)
		abs_path=os.path.abspath(str(self.output_dir))
		method_output_dir=abs_path+'/'+str(method)+'/'
		if not ( os.path.exists(method_output_dir)):

			os.mkdir(method_output_dir)
		for node in node_set:
			cmnd="./new_sir" + " "+ str(self.beta) + " "+ str(self.gamma)+ " " + str(node) + " "+ str(dataset) + " " + str(method_output_dir)+ " "  + str(self.n_iter) 
			os.system(str(cmnd))

class Util:

	def __init__(self):

		self.average_node_group_behavior=[]

	def average_results_by_method(self,method_dir,n_iterations,mode):

		os.chdir(method_dir)
		file_names=[f for f in os.listdir(method_dir)if os.path.isfile(os.path.join(method_dir, f))]
		nodes_behavior={}
		node_index=0
		max_gen_timestep=0
		# average behavior per node
		for node_file in range(0,len(file_names)):

			node_last_timesteps=[]
			node_iteration_behavior={}
			
			with open(file_names[node_file]) as f:

				if (file_names[node_file].endswith(".txt")):
					#print(str(file_names[node_file]))
					n_infected_nodes=0
					for line in f:

						line=line.rstrip()

						if "last timestep" in line:

							integers_in_last_line= [int(s) for s in line.split() if s.isdigit()]
							node_last_timesteps.append(int(integers_in_last_line[0]))

						elif "node" in line:

							integers_in_first_line= [int(s) for s in line.split() if s.isdigit()]
							n_of_iteration=int(integers_in_first_line[1])
							node_iteration_behavior[n_of_iteration]=[]
							n_infected_nodes=0

						elif not(re.search('[a-zA-Z]', line)):

							line_array=line.split(" ")
							if (mode==0):
								n_infected_nodes=int(line_array[1])
							elif (mode==1):
								n_infected_nodes+=int(line_array[1])
							#n_infected_nodes=int(line_array[1])
							node_iteration_behavior[n_of_iteration].append(n_infected_nodes)

			max_node_timestep=max(node_last_timesteps)

			if (max_gen_timestep<max_node_timestep):
				max_gen_timestep=max_node_timestep

			m=int(n_iterations)
			n=max_node_timestep+1
			analytical_node_behavior=np.zeros(shape=(m,n),dtype='f')

			for iteration, node_behavior in node_iteration_behavior.items():

				for step_value in range(0,len(node_behavior)):

					analytical_node_behavior[int(iteration)][int(step_value)]=node_behavior[int(step_value)]

				if (mode==1):

					for filler_value in range(len(node_behavior),n):

						analytical_node_behavior[int(iteration)][int(filler_value)]=node_behavior[int(step_value)]


			average_node_behavior=np.mean(analytical_node_behavior,axis=0)
			nodes_behavior[int(node_index)]=average_node_behavior.tolist()
			node_index+=1

		# average behavior of method

		m=int(len(file_names))
		n=max_gen_timestep+1
		analytical_nodes_behaviors=np.zeros(shape=(m,n),dtype='f')

		for node_index,av_node_behavior in nodes_behavior.items():
			for index in range(0,len(av_node_behavior)):
				
				analytical_nodes_behaviors[int(node_index)][int(index)]=av_node_behavior[int(index)]

			if (mode==1):

					for filler_value in range(len(av_node_behavior),n):

						analytical_nodes_behaviors[int(node_index)][int(filler_value)]=av_node_behavior[int(index)]

		self.average_node_group_behavior=np.mean(analytical_nodes_behaviors,axis=0)
		self.average_node_group_behavior=self.average_node_group_behavior.tolist()

class Plotting:

	def __init__(self,graph_dataset,sir_param_dict,labels,number_of_top_mode):

		#self.plot_type=plot_type
		self.graph_dataset=graph_dataset
		self.sir_param_dict=sir_param_dict
		self.max_timestep=0
		self.labels=labels
		self.number_of_top_mode=number_of_top_mode

	def fix_array_length(self,array,size,mode=0):

		if (len(array)<size):

			for _ in range(0,(size-len(array))):

				if (mode==0):
					array.append(0.0)
				elif (mode==1):
					array.append(array[-1])

		return array

	def plot_all(self,mode=0,file=None,type=0):

		# mode 0 for stepwise 1 for cumulative
		# type 0 for plot 1 for latex table
		# file has a value if user inputs their own metric to compare with other baselines

		graph=Graph(self.graph_dataset,self.sir_param_dict)
		print('Reading graph file...')

		graph.find_kcore()
		if (self.number_of_top_mode=='kcore'):

			self.number_of_top_mode=graph.kcore_length

		elif not isinstance(self.number_of_top_mode, int):
			raise ValueError('Wrong input for the top nodes selection, enter either an integer or type "kcore" ')

		functions=[]
		self.max_timestep=0

		for i in range(0,len(self.labels)):

			print('Calculating ' + str(labels[i]) + '...')
			top_nodes=graph.get_top_nodes(self.number_of_top_mode,str(labels[i]),file)
			path_to_sir=os.path.dirname(os.path.realpath(__file__))
			graph.run_sir(path_to_sir,top_nodes, str(labels[i]) ,self.graph_dataset)

			method=Util()
			real_output_dir=os.path.dirname(os.path.realpath(graph.output_dir))
			method.average_results_by_method(real_output_dir+'/'+ graph.output_dir+'/'+str(labels[i]),graph.n_iter,mode)

			method_result_length=len(method.average_node_group_behavior)

			if self.max_timestep<method_result_length:

				self.max_timestep=method_result_length

			functions.append(method.average_node_group_behavior)

			two_up =  real_output_dir
			os.chdir(two_up)

		for vector in range(0,len(functions)):

			functions[vector]=self.fix_array_length(functions[vector],self.max_timestep,mode)

		t=np.arange(0,self.max_timestep,1)

		os.chdir(str(graph.output_dir))

		if (type==0):
			colors = ('b', 'r', 'm', 'y' ,'c', 'g')
			markers = []
			for m in Line2D.markers:
			    try:
			        if len(m) == 1 and m != ' ':
			            markers.append(m)
			    except TypeError:
			        pass

			for c,m,f,l in zip(colors,markers,functions,self.labels):
				plt.plot(t,f , marker=m, color=c, label=l)

			plt.xlabel('Number of Steps',fontsize=18)
			plt.ylabel('Stepwise Number of Infected Nodes',fontsize=18)
			plt.title(str(graph.dataset_name))
			lgd=plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))
			
			if (mode==0):
				plt.savefig(str(graph.dataset_name)+'_stepwise_comparison.pdf',format="pdf",bbox_extra_artists=(lgd,), bbox_inches='tight')
			elif (mode==1):
				plt.savefig(str(graph.dataset_name)+'_cumulative_stepwise_comparison.pdf',format="pdf",bbox_extra_artists=(lgd,), bbox_inches='tight')

		elif (type==1):

			if (mode==0):
				res_string_file=str(graph.dataset_name)+'_stepwise_comparison.tex'
			elif (mode==1):
				res_string_file=str(graph.dataset_name)+'_cumulative_stepwise_comparison.tex'

			res_file = open(res_string_file, 'w+')
			res_file.write('\\documentclass{article}'+'\n')
			res_file.write('\\usepackage{booktabs}'+'\n')
			res_file.write('\\usepackage{caption}'+'\n')
			res_file.write('\\begin{document}'+'\n')
			res_file.write(' \\begin{table}'+'\n')
			res_file.write(' \\begin{tabular}{crrrrrr}'+'\n')
			res_file.write('\hline'+'\n')
			res_file.write(' \\multicolumn{1}{c}{}&\multicolumn{5}{c}{Time Step}'+ '\\\ '+'\n')
			res_file.write('Metric & \multicolumn{1}{c}{$~2$}  & \multicolumn{1}{c}{$~~~~4$}  & \multicolumn{1}{c}{$~~~~6$} &  \multicolumn{1}{c}{$~8$}  & \multicolumn{1}{c}{$~10$} & \multicolumn{1}{c}{Final}  \\\ '+'\n')
			res_file.write('\midrule'+'\n')

			for i in range(0,len(labels)):
				temp_list=functions[i]
				final_step=round(sum(temp_list),2)
				functions[i][2]=round(functions[i][2],2)
				functions[i][4]=round(functions[i][4],2)
				functions[i][6]=round(functions[i][6],2)
				functions[i][8]=round(functions[i][8],2)
				functions[i][10]=round(functions[i][10],2)
				res_file.write(' \\textbf{'+str(labels[i])+ '}& $\\textbf{'+ str(functions[i][2])+ '}$ & $\\textbf{'+ str(functions[i][4]) + '}$ & $\\textbf{'+ str(functions[i][6]) + '}$ & $\\textbf{'+ str(functions[i][8]) + '}$ & $\\textbf{'+ str(functions[i][10])+ '}$ & $\\textbf{'+ str(final_step)+'}$ \\\ '+'\n')
				res_file.write(' \\midrule'+'\n')

			res_file.write('\\hline'+'\n')
			res_file.write(' \\end{tabular}'+'\n')
			graph.dataset_name=graph.dataset_name.replace('_',' ')
			res_file.write(' \\caption{Spreading performance comparison for ' + str(graph.dataset_name) + ' ' + ' dataset}'+'\n')
			res_file.write('\\end{table}'+'\n')
			res_file.write('\\end{document}'+'\n')
			res_file.close()

			cmnd=['pdflatex', '-interaction', 'nonstopmode', res_string_file]
			proc = subprocess.Popen(cmnd)
			proc.communicate()
			os.unlink(res_string_file[:-4]+'.aux')
			os.unlink(res_string_file[:-4]+'.log')

def read_config_file(file):
	
	graph_dataset=''
	labels=[]
	selection_mode=0
	user_file=''
	mode=0
	type_output=0
	sir_param_dict={}


	with open(file) as f:

		for line in f:

			line=line.rstrip()

			line_array=line.split(":")
			line_array[1]=line_array[1].strip()

			if (line_array[0]=='beta'):

				sir_param_dict['beta']=float(line_array[1])

			elif (line_array[0]=='gamma'):

				sir_param_dict['gamma']=float(line_array[1])

			elif (line_array[0]=='n_iterations'):

				sir_param_dict['n_iter']=int(line_array[1])

			elif (line_array[0]=='output_dir'):

				sir_param_dict['output_dir']=str(line_array[1])

			elif (line_array[0]=='dataset'):

				graph_dataset=str(line_array[1])

			elif (line_array[0]=='labels'):

				labels=str(line_array[1]).split(',')
				
				for i in range(0,len(labels)):

					labels[i]=labels[i].strip()

			elif (line_array[0]=='selection_of_top_mode'):
				
				if ( line_array[1].isdigit() ):
					
					selection_mode=int(line_array[1])
				else:
					selection_mode=str(line_array[1])

			elif (line_array[0]=='user_input_file'):

				user_file=str(line_array[1])

			elif (line_array[0]=='mode'):

				mode=int(line_array[1])

			elif (line_array[0]=='type_output'):

				type_output=int(line_array[1])

	return sir_param_dict,graph_dataset,labels,selection_mode,user_file,mode,type_output

if __name__ == "__main__":

    if not len(sys.argv) == 2:
        print ('Wrong arguments')
    else:
    	config_file= sys.argv[1]

    sir_param_dict,graph_dataset,labels,selection_mode,user_file,mode,type_output=read_config_file(config_file)
    if not ( os.path.exists(sir_param_dict['output_dir'])):
    	os.mkdir(sir_param_dict['output_dir'])
    Plot=Plotting(graph_dataset,sir_param_dict,labels,selection_mode)
    Plot.plot_all(mode=mode,type=type_output)







