
### Title:
Informative Subgraph Extraction with Deep Reinforcement Learning for Drug-Drug Interaction Prediction

### Author & Institution
Jiancong Xie1, Wentao Wei1,2, Chi Zhang1, Jiahua Rao1*, Yuedong Yang1,3* 1School of Computer Science and Engineering, Sun Yat-sen University, China 2Pengcheng Laboratory, China 3Key Laboratory of Machine Intelligence and Advanced Computing, Sun Yat-sen University, China


### Background
Subgraph-based methods utilizing KG and domain information have achieved promising results by extracting informative subgraphs for ddi prediction.


### Gap
1. constrained by the vast and noisy nature of real-world KGs, making it challenging to identify the most informative substructures from massive candidates. 
	1. can be addressed by our claim: PK/PD specificity
	2. 他们同样没有解决，如果子图中的path不能显式存在，此时对结果的影响。仍然局限在存在，而不能进行generalize
	3. 没有考虑超过k-hop的长程节点（是否存在GNN，可以选择性的进行message passing）？
2. fail to exploit the molecular structure specificity of drugs to selectively extract relevant subgraph-based
	1. can be rewrite here - > exploit semantic information out of KG itself (LLMs); prior knowledge over this domain
	2. 分子机制在KG里面到底有多重要？对哪些pair，哪些机制更为重要？我们事实上已经回答了这个问题了。



### Methods
A reinforced-based informative subgraph extraction approach for ddi prediction
* Subgraph extraction - map > Markov Decision Process 
* A learnable structure-aware reward model that considers both the topological context from the knowledge graph and the molecular features of the drug pairs