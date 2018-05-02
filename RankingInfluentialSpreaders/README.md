==================================================================

	     Comparing Metrics to Rank Influential Spreaders

==================================================================


The model can produce plots and tables that present a thorough 
comparison among different metrics based on their
efficiency to rank influential spreaders in social networks.

The user has the possibility to give the graph dataset of their
preference as input to the model in order for the calculations to be
performed on the desired dataset. The user can also decide the metrics
whose performance will be compared. It is possible to compare Degree,
PageRank, Closeness, Eigenvector and k-core centralities. Additionally
the user can input a file which serves as a dictionary and assigns a 
specific centrality metric to each of the nodes of the dataset.

In order to simulate the spreading process, the Susceptible-Infected-
Recovered (SIR) model is used [1]. For each centrality compared, the
top ranked nodes trigger an epidemic process. The number of the top 
nodes that will be chosen to be used is defined by the user: the user 
can either input an integer (i.e., choose the top 100 nodes based on 
their Degree and PageRank centralities to trigger the epidemic process
and compare the respective performances) or choose the same number of 
nodes as the number of nodes belonging in the  k-core subgraph of the 
graph (the latter assigns the same maximum k-core centrality to a large
group of nodes and it is strongly advised to use the same number of top
nodes when comparing any centrality with the k-core one). Each spreading
process is triggered by each individual node of the group of nodes chosen
by the respective centrality and the performance reported for every metric
is actually the average behavior of the whole group. 

///////////////////////////////////////////////////////////////////////////////////////

To run it type:

python spreading_experiments.py /.../config_file.txt

The configuration file should contain the following parameters in the 
following format:

dataset: /path/to/graph/dataset.txt (The dataset should be in the form of
	      an edgelist- the two node ids should be separated by a space)

labels: metric1, metric2,...,metric6 (The metrics to be compared separated
		by ','. You can choose among Degree, Pagerank, Closeness, Eigenvector
		and k-core centralities [2,3]. Write degree, pagerank, closeness, 
		eigenvector or kcore respectively. You can also input a metric of your
		preference-you should keep in minf that to do that you have to input 
		the name of your metric in this field and also completing the 
		'user_input_file' field.)

selection_of_top_mode: int/kcore (You can either input an integer to choose how
						many nodes having top values will be chosen from each 
						centrality compared either input 'kcore' which indicates 
						that the number of nodes that will be chosen is the number
						of nodes in the maximum k-core subgraph of the dataset.)

user_input_file: /path/to/user/input/dictionary/file (If you want to compare some
				 of the baseline metrics with a metric of your choice that is not 
				 one of the baselines, input a file that serves as dictionary. The
				 file should have the following format: two columns separated by a
				 space, the first depicts the node id and the second its respective
				 value of the chosen centrality.)

mode: 0/1 (The output provides a comparison of the chosen metrics by depicting
          either the number of nodes being infected/influenced in the spreading 
          process (0) at each time-step either the cumulative number of nodes 
		  infected/influenced at each time-step (1).)

type: 0/1 (The model outputs either a plot(0) or a table(1) (both .tex and .pdf are 
           being produced). The output depicts a per-step comparison of the spreading
           performance triggered by the nodes chosen by the different centralities.)

beta: float (Input the beta probability of the SIR model (i.e., the probability that a
			 node passes from the Susceptible to the Infected state if one of its 
			 neighbors is already infected). Choose a value close to the epidemic 
			 threshold of the dataset.)

gamma: float (Input the beta probability of the SIR model. Usually set to 0.8 or 1.0)

n_iterations: int (Input the number of times you want the SIR process to be triggered by
				   each of the nodes of the group. Reminder: the SIR model is a 
				   probabilistic model so it is advised that for the correct behavior of
				   each node to be depicted, the process is repeated multiple times. Usually
				   set to 100.)

output_dir: /path/to/directory/of/results (The behavior of the nodes for every chosen metric
			will be stored in .txt files. Also the resulting file will be saved (.pdf format
			for the case of a plot and both .tex and .pdf format for the case of the table).)

///////////////////////////////////////////////////////////////////////////////////////////



[1] Kermack, W. O. & McKendrick, A. A contribution to the Mathematical 
theory of epidemics. Proceedings of the Royal Society of London 115(772),
700–721 (1927).

[2] Koschützki, D., Lehmann, K. A., Peeters, L., Richter, S., Tenfelde-Podehl,
D., & Zlotowski, O. (2005). Centrality indices. In Network analysis (pp. 16-61).
Springer Berlin Heidelberg.

[3] Batagelj, V., & Zaversnik, M. (2003). An O (m) algorithm for cores decomposition 
of networks. arXiv preprint cs/0310049.







