import pandas as pd
from collections import defaultdict
import warnings

import neat
import multiprocessing
import os
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import KFold
from random import randint
import os
import random

kf = KFold(n_splits = 10, shuffle=True)

train_df = pd.read_csv('../ecoli.data', header=None, sep='\s+', error_bad_lines=False)

XX = train_df.iloc[:,1:8].to_numpy() 
yy = train_df.iloc[:,8:].to_numpy().reshape(len(train_df))

yy[yy == 'cp'] = 0
yy[yy == 'im'] = 1
yy[yy == 'imS'] = 2
yy[yy == 'imL'] = 3
yy[yy == 'imU'] = 4
yy[yy == 'om'] = 5
yy[yy == 'omL'] = 6
yy[yy == 'pp'] = 7

necoc = 15
generations = 200

nfold = 1

test_acc_list = []
train_acc_list = []
train_class_list = []
node_list = []
connection_list = []

for train_index, test_index in kf.split(XX):
    print("Fold:{}".format(nfold))
    nfold += 1
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = XX[train_index], XX[test_index]
    y_train, y_test = yy[train_index], yy[test_index]
    
    numbers = [*range(8)] # number 2 > number 1
    number_of_classification = 8
    
    num_train = X_train.shape[0]
    num_test = X_test.shape[0]
    
   
    X = X_train
    y = y_train
    testX = X_test
    testy = y_test

    numbers = [*range(8)] # number 2 > number 1
    number_of_classification = 8


    list_y = y.tolist()
    digits_indexes = []
    for digit in numbers:
        li = [i for i in range(len(list_y)) if list_y[i] == digit]
        digits_indexes.extend(li)

    samplesize = X[digits_indexes].shape[0]
    x_inputs =[tuple(c) for c in X[digits_indexes].tolist()]
    x_outputs = [tuple(c) for c in y[digits_indexes].reshape(samplesize,1).tolist()]

    test_list_y =y_test.tolist()
    digits_indexes = []
    for digit in numbers:
        li = [i for i in range(len(test_list_y)) if test_list_y[i] == digit]
        digits_indexes.extend(li)

    test_x_inputs =[tuple(c) for c in testX[digits_indexes].tolist()]
    test_x_outputs = [tuple(c) for c in y_test[digits_indexes].reshape(num_test,1).tolist()]
    
    def softmax(x, axis=1):
        # 计算每行的最大值
        row_max = x.max(axis=axis)

        # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
        row_max=row_max.reshape(-1, 1)
        x = x - row_max

        # 计算e的指数次幂
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s

    def get_winner(class1):

        numbers = [*range(8)] # number 2 > number 1
        number_of_classification = len(numbers)
        #number_of_sampling = 100

        list_y = y.tolist()
        digits_indexes = []
        for digit in numbers:
            li = [i for i in range(len(list_y)) if list_y[i] == digit]
            digits_indexes.extend([li])


        class2 = []

        class1_indexes = []
        class2_indexes = []

        for i in numbers:
            if i in class1:
                class1_indexes.extend(digits_indexes[i])
            else:
                class2_indexes.extend(digits_indexes[i])
                class2.append(i)

        class1_label = [1] * len(class1_indexes)
        class2_label = [0] * len(class2_indexes)

        #print("number of class1:{}".format(len(class1_label)))
        #print("number of class2:{}".format(len(class2_label)))

        x_inputs =[tuple(c) for c in X[class1_indexes].tolist()] + [tuple(c) for c in X[class2_indexes].tolist()]
        x_outputs = [tuple([c]) for c in class1_label + class2_label]


        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                net = neat.nn.FeedForwardNetwork.create(genome, config)

                outputs = []
                for xi in x_inputs:
                    output = net.activate(xi)
                    outputs.append(output)


                px_outputs = softmax(np.array(outputs).reshape(samplesize, 2), axis=1)
                # the index of maximum in each line
                pred_outputs = np.argmax(px_outputs, axis = 1)
                real_outputs = np.array(x_outputs).reshape(samplesize,)

                acc = np.sum(pred_outputs == real_outputs)/samplesize

                genome.fitness = acc

        def run(config_file):
            # Load configuration.
            config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                config_file,
            )

            # Create the population, which is the top-level object for a NEAT run.
            p = neat.Population(config)

            # add a stdout reporter to show progress in the terminal
            #reporter = neat.StdOutReporter(False)
            #p.add_reporter(reporter)
            #stats = neat.StatisticsReporter()
            #p.add_reporter(stats)
            #checkpointer = neat.Checkpointer(100)
            #p.add_reporter(checkpointer)
            # Run for up to 300 generations.
            winner = p.run(eval_genomes, generations)

            return winner

        local_dir = os.getcwd()
        config_path = os.path.join(local_dir, "config-feedforward-ecoli2")
        winner = run(config_path)

        return winner
    
    while True:
        number_codes = []
        for i in range(2**(8-1),2**8-1):
            for j in bin(i).split('b')[1]:
                number_codes.append(int(j))
        number_codes = np.array(number_codes).reshape(2**(8-1) -1,8).T

        slice = random.sample(range(2**(8-1) -1),necoc)

        number_codes = number_codes[:,slice].tolist()

        classes = []
        for i in range(necoc):
            code_list = np.array(number_codes).T[i]
            classes.append(np.where(code_list==1)[0].tolist())

        matrix_errors = []
        for i in range(8-1):
            for j in range(i+1,8):
                matrix_errors.append(necoc - np.sum(np.array(number_codes[i]) == np.array(number_codes[j])))
        min_error = np.min(matrix_errors)
        if min_error != 0:
            break
    print(min_error)

    
    winner_list = []
    for class1 in classes:
        winner = get_winner(class1)
        winner_list.append(winner)
    
    local_dir = os.getcwd()
    config_path = os.path.join(local_dir, "config-feedforward-ecoli2")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    def get_pred_real(i, j):
        winner_net = neat.nn.FeedForwardNetwork.create(winner_list[i], config)

        numbers = [*range(8)]
        class1 = classes[i]
        class2 = []
        class1_indexes = []
        class2_indexes = []

        test_X = testX[j]
        test_y = testy[j]

        list_y = [test_y]
        digits_indexes = []
        for digit in numbers:
            li = [i for i in range(len(list_y)) if list_y[i] == digit]
            digits_indexes.extend([li])

        for i in numbers:
            if i in class1:
                class1_indexes.extend(digits_indexes[i])
            else:
                class2_indexes.extend(digits_indexes[i])
                class2.append(i)

        class1_label = [1] * len(class1_indexes)
        class2_label = [0] * len(class2_indexes)


        testsamplesize = 1
        test_x_inputs =[tuple(test_X)]
        test_x_outputs = [tuple([c]) for c in class1_label + class2_label]

        outputs = []
        for xi in test_x_inputs:
            output = winner_net.activate(xi)
            outputs.append(output)

        px_outputs = softmax(np.array(outputs).reshape(testsamplesize, 2), axis=1)
        # the index of maximum in each line
        pred_outputs = np.argmax(px_outputs, axis = 1)
        real_outputs = np.array(test_x_outputs).reshape(testsamplesize,)


        return [pred_outputs, real_outputs]
    
    pred_value = []
    error_list = []
    
    testsamplesize = num_test
    for j in range(testsamplesize):
        pred = []
        for i in range(necoc):
            [pred_outputs, real_outputs] = get_pred_real(i, j)
            #print(pred_outputs, real_outputs)
            pred.append(pred_outputs)
        #print(np.array(pred).T)

        error = []
        for i in range(8):
            error.append(necoc - np.sum(number_codes[i] == np.array(pred).T))
        #print(error)

        pred_value.append(np.where(error==np.min(error)))
        error_list.append(np.min(error))
        #print(np.where(error==np.min(error)) )
        
    list_P = []
    for i in pred_value:
        if (len(i[0])) == 1:
            list_P.append(i[0][0])
        else:
            random_pick = randint(0, len(i[0])-1)
            list_P.append(i[0][random_pick])
    print("Test accuracy:{}".format(np.sum(list_P == testy)/testsamplesize))
    test_acc_list.append(np.sum(list_P == y_test)/num_test)


    
    cm = confusion_matrix(np.array(list_P), y_test.tolist())
    print(cm)
    
    def get_pred_real(i, j):
        winner_net = neat.nn.FeedForwardNetwork.create(winner_list[i], config)

        numbers = [*range(8)]
        class1 = classes[i]
        class2 = []
        class1_indexes = []
        class2_indexes = []

        test_X = X[j]
        test_y = y[j]

        list_y = [test_y]
        digits_indexes = []
        for digit in numbers:
            li = [i for i in range(len(list_y)) if list_y[i] == digit]
            digits_indexes.extend([li])

        for i in numbers:
            if i in class1:
                class1_indexes.extend(digits_indexes[i])
            else:
                class2_indexes.extend(digits_indexes[i])
                class2.append(i)

        class1_label = [1] * len(class1_indexes)
        class2_label = [0] * len(class2_indexes)


        testsamplesize = 1
        test_x_inputs =[tuple(test_X)]
        test_x_outputs = [tuple([c]) for c in class1_label + class2_label]

        outputs = []
        for xi in test_x_inputs:
            output = winner_net.activate(xi)
            outputs.append(output)

        px_outputs = softmax(np.array(outputs).reshape(testsamplesize, 2), axis=1)
        # the index of maximum in each line
        pred_outputs = np.argmax(px_outputs, axis = 1)
        real_outputs = np.array(test_x_outputs).reshape(testsamplesize,)


        return [pred_outputs, real_outputs]


    pred_value = []
    error_list = []

    for j in range(samplesize):
        pred = []
        for i in range(necoc):
            [pred_outputs, real_outputs] = get_pred_real(i, j)
            #print(pred_outputs, real_outputs)
            pred.append(pred_outputs)
        #print(np.array(pred).T)

        error = []
        for i in range(8):
            error.append(necoc - np.sum(number_codes[i] == np.array(pred).T))
        #print(error)

        pred_value.append(np.where(error==np.min(error)))
        error_list.append(np.min(error))
        #print(np.where(error==np.min(error)) )

    list_P = []
    for i in pred_value:
        if (len(i[0])) == 1:
            list_P.append(i[0][0])
        else:
            random_pick = randint(0, len(i[0])-1)
            list_P.append(i[0][random_pick])
    print("Training accuracy:{}".format(np.sum(list_P == y)/samplesize))

    winner_fitness = []
    for winner in winner_list:
        winner_fitness.append(winner.fitness)
    print("Training classifier accuracy:{}".format(np.mean(winner_fitness)))

    train_acc_list.append(np.sum(list_P == y)/num_train)
    train_class_list.append(np.mean(winner_fitness))
    
    # program to check if there is exist a path between two vertices 
# of a graph 


    #This class represents a directed graph using adjacency list representation 
    class Graph: 

        def __init__(self,vertices): 
            self.V= vertices #No. of vertices 
            self.graph = defaultdict(list) # default dictionary to store graph 

        # function to add an edge to graph 
        def addEdge(self,u,v): 
            self.graph[u].append(v) 

        # Use BFS to check path between s and d 
        def isReachable(self, s, d): 
            # Mark all the vertices as not visited 
            visited =[False]*(self.V) 

            # Create a queue for BFS 
            queue=[] 

            # Mark the source node as visited and enqueue it 
            queue.append(s) 
            visited[s] = True

            while queue: 

                #Dequeue a vertex from queue 
                n = queue.pop(0) 

                # If this adjacent node is the destination node, 
                # then return true 
                if n == d: 
                    return True

                # Else, continue to do BFS 
                for i in self.graph[n]: 
                    if visited[i] == False: 
                        queue.append(i) 
                        visited[i] = True
            # If BFS is complete without visited d 
            return False


    def findAllPath(graph,start,end,path=[]):
        path = path +[start]
        if start == end:
            return [path]

        paths = [] #存储所有路径    
        for node in graph[start]:
            if node not in path:
                newpaths = findAllPath(graph,node,end,path) 
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    numbers = [*range(2)]
    number_of_classification = len(numbers)

    list_nodes_number = []
    list_connection_number = []

    for winner in winner_list:
        used_nodes = list(winner.nodes.keys())
        # create a dict for mapping 
        l1 = numbers + config.genome_config.input_keys + used_nodes[number_of_classification:]
        l2 = range(len(l1))
        dict_nodes = dict(zip(l1, l2))

        ### 将节点加入图
        g = Graph(len(l1))# inputs + outputs + used  
        # add connections in the graph
        for cg in winner.connections.values():
            if cg.enabled:
                g.addEdge(dict_nodes[cg.key[0]],dict_nodes[cg.key[1]])

        v = numbers
        list_connections = []
        for vi in v:
            for u in range(number_of_classification, len(config.genome_config.input_keys) + number_of_classification):
                list_connections.append(g.isReachable(u, vi))

        nodes_inputs = range(number_of_classification, number_of_classification+ 9)
        nodes_outputs = range(number_of_classification)
        nodes_mid = range(number_of_classification + 9, len(dict_nodes))

        # 找到所有输入和输出连接的路
        all_path = []
        for u in nodes_inputs:
            for v in nodes_outputs:
                path = findAllPath(g.graph, u, v)
                if path:
                    all_path = all_path + path

        # 得到最长路，以及每个节点在哪个层级
        max_length = max([len(x) for x in all_path])
        nodes_tuples_list = []
        for path in all_path:
            #print(path)
            for node in path:
                nodes_tuples_list.append([node, path.index(node)])

        # 确定节点的层级（消除重复）       
        nodes_tuples_fixed_list = []
        for index in range(max_length):
            for nodes in nodes_tuples_list:
                node, node_index = nodes[0], nodes[1]
                if node_index == index: 
                    if nodes not in nodes_tuples_fixed_list:
                        nodes_tuples_fixed_list.append(nodes)


        # 所有有连接的节点
        all_used_nodes = []
        for node in nodes_tuples_fixed_list:
            if node[0] not in all_used_nodes:
                all_used_nodes.append(node[0])

        # 确定节点的层级（最终确定）
        nodes_tuples_fixed_list_final = []

        for node in all_used_nodes:
            if node in nodes_inputs:
                nodes_tuples_fixed_list_final.append([node, 0])
            elif node in nodes_outputs:
                nodes_tuples_fixed_list_final.append([node, max_length - 1])
            else:
                list_node = []
                for nodes in nodes_tuples_fixed_list:
                    if node == nodes[0]:
                        list_node.append(nodes)
                layer = 0
                for ele in list_node:
                    # 删除重复项，取最大
                    if ele[1] > layer:
                        layer = ele[1]
                nodes_tuples_fixed_list_final.append([node,layer])


        # 每个层级的节点合数
        length_of_layers = []
        number_of_nodes = 0
        for index in range(max_length):
            for nodes in nodes_tuples_fixed_list_final:
                node, node_index = nodes[0], nodes[1]
                if node_index == index: 
                    number_of_nodes += 1
            length_of_layers.append(number_of_nodes)

        length_of_each_layer = []
        length_of_each_layer.append(length_of_layers[0]) 
        for i in range(1, max_length):
            length_of_each_layer.append(length_of_layers[i] - length_of_layers[i-1])

        # 输出每个层级的节点个数
        #print("length of each layers:", length_of_each_layer)

        # 所有端到端的路
        all_path_side2side = []
        for path in all_path:
            if len(path) == 2:
                all_path_side2side.append(path)
            else:
                for i in range(len(path)-1):
                    all_path_side2side.append([path[i],path[i+1]])

        # 定义节点到绘图的字典
        dict_nodes_graph = dict()
        count_number_layer = [0] * max_length
        for nodes in nodes_tuples_fixed_list_final:
            node, layer = nodes
            dict_nodes_graph[node] = [count_number_layer[layer], layer]
            count_number_layer[layer] += 1
        list_nodes_number.append(np.sum(length_of_each_layer))
        list_connection_number.append(len(all_path_side2side))
        #print("Number of nodes:{} Number of connections:{}".format(np.sum(length_of_each_layer),len(all_path_side2side)))
    print("Total nodes:{} Total connections:{}".format(np.sum(list_nodes_number),np.sum(list_connection_number)))
    node_list.append(np.sum(list_nodes_number))
    connection_list.append(np.sum(list_connection_number))    
    
print(test_acc_list,train_acc_list,train_class_list,node_list,connection_list)
print(np.mean(test_acc_list),np.mean(train_acc_list),np.mean(train_class_list),np.mean(node_list),np.mean(connection_list))