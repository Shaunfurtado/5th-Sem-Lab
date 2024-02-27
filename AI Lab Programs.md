# 1.Astar

``` python
def aStarAlgo(start_node,stop_node):
    open_set=set(start_node)
    closed_set=set()
    g = {}
    parents = {}
    g[start_node] = 0
    parents[start_node] = start_node
    
    while len(open_set) > 0:
        n=None
        for v in open_set:
            if n == None or g[v] + heuristic(v) < g[n] +  heuristic(n):
                n = v
                
        if n==stop_node or Graph_nodes[n] == None:
            pass
        else:
            for (m,  weight) in get_neighbors(n):
                
                if m not in open_set and m not in closed_set:
                    open_set.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                else:
                    if g[m] > g[n]+weight:
                        g[m]=g[n]+weight
                        parents[m]=n 
                        if m in closed_set:
                            closed_set.remove(m)
                            open_set.add(m)
        if n==None:
            print('path does not exist')
            return None
        if n == stop_node:
            path=[]
            while parents[n] !=n:
                path.append(n)
                n=parents[n]
            path.append(start_node)
            path.reverse()
            print('Path found: {}'.format(path))
            return path
        open_set.remove(n)
        closed_set.add(n)
    print('path does not exist!')
    return None

def get_neighbors(v):
            if v in Graph_nodes:
                return Graph_nodes[v] 
            else:
                return None
            
def heuristic(n):
    H_dist={
        'A':11,
        'B':6,
        'C':99,
        'D':1,
        'E':7,
        'G':0
    }     
    return H_dist[n]

Graph_nodes={
    'A':[('B',2),('E',3)],
    'B':[('G',9),('C',1)],
    'E':[('D',6)],
    'D':[('G',1)]
}
aStarAlgo('A','G')
```
# 2.Best First Search
``` python
import heapq
Graph_nodes = { 
   'S': ['A','B'], 
   'A': ['C','D'], 
   'B': ['E','F'], 
   'E': ['H'],
   'F': ['I','G']
   }
def heuristic(n):
    H_dist = { 
        'S': 13,
        'A': 12,
        'B': 4,
        'C': 7,
        'D': 3,
        'E': 8,
        'F': 2,
        'H': 4,
        'I': 9,
        'G': 0
     } 
    return H_dist[n] 

def best_first_search(graph, start, goal):
    open = [(heuristic(start), start)]
    closed = {}
    closed[start] = None
    
    while open:
        _,peak_node = heapq.heappop(open)
        
        if peak_node == goal:
            break
            
        for neighbor in graph[peak_node]:
            if neighbor not in closed:
                heapq.heappush(open, (heuristic(neighbor), neighbor))
                closed[neighbor] = peak_node
    
    return closed
start_node = 'S'
goal_node = 'G'
closed = best_first_search(Graph_nodes, start_node, goal_node)
print("Closed list",closed)
node = goal_node
path = [node]

while node != start_node:
    node = closed[node]
    path.append(node)
    
path.reverse()
print("BFS path from",start_node,"to",goal_node,":",path)
```
# 3.AOStar
``` python
class Graph:
    def __init__(self, graph, heuristicNodeList, startNode):
        self.graph = graph
        self.H=heuristicNodeList
        self.start=startNode
        self.parent={}
        self.status={}
        self.solutionGraph={}
     
    def applyAOStar(self):        
        self.aoStar(self.start, False)

    def getNeighbors(self, v):    
        return self.graph.get(v,'')
    
    def getStatus(self,v):        
        return self.status.get(v,0)
    
    def setStatus(self,v, val):    
        self.status[v]=val
    
    def getHeuristicNodeValue(self, n):
        return self.H.get(n,0)     
 
    def setHeuristicNodeValue(self, n, value):
        self.H[n]=value            
        
    
    def printSolution(self):
        print("FOR GRAPH SOLUTION, TRAVERSE THE GRAPH FROM THE START NODE:",self.start)
        print("------------------------------------------------------------")
        print(self.solutionGraph)
        print("------------------------------------------------------------")
    
    def computeMinimumCostChildNodes(self, v):  
        minimumCost=0
        costToChildNodeListDict={}
        costToChildNodeListDict[minimumCost]=[]
        flag=True
        for nodeInfoTupleList in self.getNeighbors(v):  
            cost=0
            nodeList=[]
            for c, weight in nodeInfoTupleList:
                cost=cost+self.getHeuristicNodeValue(c)+weight
                nodeList.append(c)
            
            if flag==True:                      
                costToChildNodeListDict[minimumCost]=nodeList      
                flag=False
            else:                                 
                if minimumCost>cost:
                    minimumCost=cost
                    costToChildNodeListDict[minimumCost]=nodeList  
                
              
        return minimumCost, costToChildNodeListDict[minimumCost]   
    
    def aoStar(self, v, backTracking):     
                print("HEURISTIC VALUES  :", self.H)
                print("SOLUTION GRAPH    :", self.solutionGraph)
                print("PROCESSING NODE   :", v)
                print("-----------------------------------------------------------------------------------------")
        
                if self.getStatus(v) >= 0:       
                    minimumCost, childNodeList = self.computeMinimumCostChildNodes(v)
                    self.setHeuristicNodeValue(v, minimumCost)
                    self.setStatus(v,len(childNodeList))
            
                    solved=True                   
                    for childNode in childNodeList:
                        self.parent[childNode]=v
                        if self.getStatus(childNode)!=-1:
                            solved=solved & False
            
                    if solved==True:             
                        self.setStatus(v,-1)    
                        self.solutionGraph[v]=childNodeList

                    if v!=self.start:          
                        self.aoStar(self.parent[v], True)  
                
                    if backTracking==False:     
                        for childNode in childNodeList:  
                            self.setStatus(childNode,0)  
                            self.aoStar(childNode, False) 

h1 = {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1, 'T': 3}
graph1 = {
    'A': [[('B', 1), ('C', 1)], [('D', 1)]],
    'B': [[('G', 1)], [('H', 1)]],
    'C': [[('J', 1)]],
    'D': [[('E', 1), ('F', 1)]],
    'G': [[('I', 1)]]   
}

G1= Graph(graph1, h1, 'A')
G1.applyAOStar() 
```
# 4.ID3
``` python
import pandas as pd
df_tennis = pd.read_csv('4.csv')
print("\n Given Play Tennis Data Set:\n\n", df_tennis)

def entropy(probs):  
    import math
    return sum( [-prob*math.log(prob, 2) for prob in probs] )

def entropy_of_list(a_list):  
    from collections import Counter
    cnt = Counter(x for x in a_list)   
    num_instances = len(a_list)*1.0   
    probs = [x / num_instances for x in cnt.values()]  
    return entropy(probs) 

def information_gain(df, split_attribute_name, target_attribute_name, trace=0):
    df_split = df.groupby(split_attribute_name)
    nobs = len(df.index) * 1.0
    df_agg_ent = df_split.agg({target_attribute_name : [entropy_of_list, lambda x: len(x)/nobs] })[target_attribute_name]
    df_agg_ent.columns = ['Entropy', 'PropObservations']
    new_entropy = sum( df_agg_ent['Entropy'] * df_agg_ent['PropObservations'] )
    old_entropy = entropy_of_list(df[target_attribute_name])
    return old_entropy - new_entropy

def id3(df, target_attribute_name, attribute_names, default_class=None):
    from collections import Counter
    cnt = Counter(x for x in df[target_attribute_name])
    print(cnt)
    if len(cnt) == 1:
        print(len(cnt))
        return next(iter(cnt))
    elif df.empty or (not attribute_names):
        return default_class  
    else:      
        default_class = max(cnt.keys()) 
        gainz = [information_gain(df, attr, target_attribute_name) for attr in attribute_names] 
        print("Gain=",gainz)
        index_of_max = gainz.index(max(gainz)) 
        best_attr = attribute_names[index_of_max]
        print("Best Attribute:",best_attr)
        tree = {best_attr:{}} 
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]
        
        
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset,
                        target_attribute_name,
                        remaining_attribute_names,
                        default_class)
            tree[best_attr][attr_val] = subtree
        return tree


attribute_names = list(df_tennis.columns)
print("List of Attributes:", attribute_names) 
attribute_names.remove('PlayTennis')  
print("Predicting Attributes:", attribute_names)


from pprint import pprint
tree = id3(df_tennis,'PlayTennis',attribute_names)
print("\n\nThe Resultant Decision Tree is :\n")
pprint(tree)
```
# 5.Non-parametric Locally Weighted Regression
``` python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def kernel(point, xmat, k):
    m, n = np.shape(xmat)
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diff = point - X[j]
        weights[j,j] = np.exp(diff*diff.T/(-2.0*k**2))
    return weights

 
def localWeight(point, xmat, ymat, k):
    wei = kernel(point, xmat, k)
    W = (X.T*(wei*X)).I*(X.T*(wei*ymat.T))
    return W

def localWeightRegression(xmat, ymat, k):
    m, n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i] * localWeight(xmat[i], xmat,ymat, k)
    return ypred

def graphPlot(X, ypred):
    sortindex = X[:,1].argsort(0)
    xsort = X[sortindex][:,0]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(bill, tip, color='green')
    ax.plot(xsort[:,1], ypred[sortindex], color = 'red', linewidth = 5)
    plt.xlabel('Total bill')
    plt.ylabel('Tip')
    plt.show()

data = pd.read_csv('data9.csv')
bill = np.array(data.total_bill)
tip = np.array(data.tip)

mbill = np.mat(bill)
mtip = np.mat(tip)
m = np.shape(mbill)[1]
one = np.mat(np.ones(m))
X = np.hstack((one.T, mbill.T))

ypred = localWeightRegression(X, mtip, 5)
graphPlot(X, ypred)
```
# 6.Naïve Bayesian classifier
``` python
import csv, random, math


def loadcsv(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    print(trainSize)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


def seperateByClass(dataset):
    seperated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in seperated):
            seperated[vector[-1]] = []
        seperated[vector[-1]].append(vector)
    return seperated


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stddev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stddev(attribute)) for attribute in zip(*dataset)]

    del summaries[-1]
    return summaries


def summarizeByClass(dataset):
    seperated = seperateByClass(dataset)
    summaries = {}
    for classValue, instances in seperated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def calculateProbablity(x, mean, stddev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stddev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stddev)) * exponent


def calculateClassProbablities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stddev = classSummaries[i]
            x = inputVector[i]
            print("**", mean, stddev, x)
            probabilities[classValue] *= calculateProbablity(x, mean, stddev)
            print("Individual probability", probabilities[classValue])
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbablities(summaries, inputVector)
    print("@@@@", probabilities)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if (testSet[i][-1] == predictions[i]):
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def NaiveBayes():
    filename = 'data6.csv'
    splitRatio = 0.8
    dataset = loadcsv(filename)

    print("\nThe length of the Data Set :", len(dataset))
    print("\nThe Data Set Splitting into Training and Testing \n")
    trainingSet, testSet = splitDataset(dataset, splitRatio)

    print("\nNumber of Rows in Training Set:{0} rows".format(len(trainingSet)))
    print("\nNumber of Rows in Testing Set:{0} rows".format(len(testSet)))
    print("\nFirst Five Rows of Training Set:\n")
    for i in range(0, 5):
        print(trainingSet[i], "\n")
    print("\nFirst Five Rows of Testing Set:\n")
    for i in range(0, 3):
        print(testSet[i], "\n")

    summaries = summarizeByClass(trainingSet)
    print("\nModel Summaries:\n", summaries)

    predictions = getPredictions(summaries, testSet)
    print("\n Predictions:\n",predictions)

    accuracy = getAccuracy(testSet, predictions)
    print("\nAccuracy: {0}%".format(accuracy))
import csv,random,math
NaiveBayes()
```
# 7.k-Nearest Neighbor
``` python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets

iris=datasets.load_iris()

x = iris.data
y = iris.target
print ('sepal-length', 'sepal-width', 'petal-length', 'petal-width')
print(x)
print('class: 0-Iris-Setosa, 1- Iris-Versicolour, 2- Iris-Virginica')
print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)

y_pred=classifier.predict(x_test)

print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred))
print('Accuracy Metrics')
print(classification_report(y_test,y_pred))
```
# 8.Artificial Neural Network by implementing the Back propagation algorithm
``` python
import numpy as np  

# Get the data
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float) 
y = np.array(([92], [86], [89]), dtype=float)       
X = X/np.amax(X,axis=0) 
y = y/100

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)

epoch=5000              
lr=0.1                  
inputlayer_neurons = 2  
hiddenlayer_neurons = 3 
output_neurons = 1      
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons)) 
bh=np.random.uniform(size=(1,hiddenlayer_neurons))                  
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))   
bout=np.random.uniform(size=(1,output_neurons))

for i in range(epoch):
    
    hinp1=np.dot(X,wh)             
    hinp=hinp1 + bh                
    hlayer_act = sigmoid(hinp)      
    outinp1=np.dot(hlayer_act,wout) 
    outinp= outinp1+ bout
    output = sigmoid(outinp)

    EO = y-output                          
    outgrad = derivatives_sigmoid(output)  
    d_output = EO* outgrad                  
    EH = d_output.dot(wout.T)              
    hiddengrad = derivatives_sigmoid(hlayer_act) 
    d_hiddenlayer = EH * hiddengrad
    wout += hlayer_act.T.dot(d_output) *lr  
    wh += X.T.dot(d_hiddenlayer) *lr
    
print("Input: \n" + str(X)) 
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)
```
# 9.EM Algorithm
``` python
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.mixture import GaussianMixture 
from sklearn.cluster import KMeans 
data = pd.read_csv('data/ex.csv') 
f1 = data['V1'].values 
f2 = data['V2'].values 
X = np.array(list(zip(f1, f2))) 
print("x: ", X) 
print('Graph for whole dataset') 
plt.scatter(f1, f2, c='black') 
plt.show() 
kmeans = KMeans(2) 
labels = kmeans.fit(X).predict(X) 
print("labels for kmeans:", labels) 
print('Graph using Kmeans Algorithm') 
plt.scatter(f1, f2, c=labels) 
centroids = kmeans.cluster_centers_ 
print("centroids:", centroids) 
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='red') 
plt.show() 
gmm = GaussianMixture(2) 
labels = gmm.fit(X).predict(X) 
print("Labels for GMM: ", labels) 
print('Graph using EM Algorithm') 
plt.scatter(f1, f2, c=labels) 
plt.show()
```
