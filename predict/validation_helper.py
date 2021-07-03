'''initialize FLAGS parameters'''

import datetime
import os

import numpy as np


def init_Flags(FLAGS, train_slots, test_slots, list_cluster_users, list_cluster_services):
    
    FLAGS.num_batches = len(train_slots)
    FLAGS.num_samples = len(test_slots)
    
    if FLAGS.typePrediction == 'user':
        
        FLAGS.layer_sizes = FLAGS.userService_layer_sizes
        FLAGS.batch_size = 142
        FLAGS.input_dim = 4500  
        
        if FLAGS.userCluster == True:
            FLAGS.batch_size = len(list_cluster_users)
                        
        if FLAGS.serviceCluster == True:
            FLAGS.input_dim = len(list_cluster_services)
         
    if FLAGS.typePrediction == 'service':
        
        FLAGS.layer_sizes = FLAGS.serviceUser_layer_sizes
        FLAGS.batch_size = 4500
        FLAGS.input_dim = 142
        
        if FLAGS.serviceCluster == True:
            FLAGS.batch_size = len(list_cluster_services)
             
        if FLAGS.userCluster == True:
            FLAGS.input_dim = len(list_cluster_users)

def get_tensorbordfolder(FLAGS, index):
    noise = '_noise=0'
    if FLAGS.model == 1:
        noise = '_noise=' + str(FLAGS.noise)
    if FLAGS.qosMetric == 0:
        qos = "rt"
    else:
        qos = "tp"
        
    name = qos+'_all_cluster_model=' + str(FLAGS.model) + '_type=' + str(FLAGS.typePrediction) + '_density='+ str(FLAGS.training_density) +'_op=' + str(FLAGS.Optimizer) + noise + '_date='
    return str(index) + '_' + name  

def writeFile(FLAGS, text, trace):
    noise = '_noise=0'
    if FLAGS.model == 1:
        noise = '_noise=' + str(FLAGS.noise)
    if FLAGS.qosMetric == 0:
        qos = "rt"
    else:
        qos = "tp"
    name = qos+'_all_cluster_model=' + str(FLAGS.model) + '_type=' + str(FLAGS.typePrediction) + '_density='+ str(FLAGS.training_density) +'_op=' + str(FLAGS.Optimizer) + noise + '_date='
 
    msg = text + '\n' + FLAGS.flags_into_string() 
     
    now = datetime.datetime.now()
     
    file_name = name + str(now.year) + '_' + str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + str(now.minute) + str(now.second)
 
    path = '../results/' + file_name + '.txt'
    path = os.path.join(os.path.dirname(__file__), path)
    file = open(path, "a")
    file.write(msg)
    file.close
    
    filetrace = os.path.join(os.path.dirname(__file__),'../results/' + file_name +'_trace')
    np.save(filetrace, trace)
     
def get_train_test_slots_lists(FLAGS):
    #A supprimer 
    list_test = []
    list_train = []
    if FLAGS.training_density == 0.2:
        list_train = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                               [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
                               [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
                               [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
                               [52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
                            ])
 
        
        list_test = np.array([[13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63], 
                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63], 
                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63], 
                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63], 
                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
                            ])
        return list_train, list_test 
        print('ttttt')
        
    if FLAGS.training_density == 0.5:
        list_train = np.array([[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                               [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
                               [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57],
                               [0 , 1 , 2 , 3 , 4 , 5 , 6 , 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
                               [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
                            ])
 
        list_test = np.array([[32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63], 
                              [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10, 11, 12, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63], 
                              [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 58, 59, 60, 61, 62, 63], 
                              [7 , 8 , 9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38], 
                              [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
                            ])
        return list_train, list_test 
        print('ttttt')
        
    if FLAGS.training_density == 0.8:
        list_train = np.array([[13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63], 
                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63], 
                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63], 
                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63], 
                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
                            ])
        
        list_test = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                               [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
                               [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
                               [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
                               [52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
                            ])
        return list_train, list_test 
        print('ttttt')
    #Fin a supprimer
    
    pos_window = 0
    list_test = []
    list_train = []
     
    slot = FLAGS.slots
     
    end = False
     
    while not end :
        size_window = slot * (1 - FLAGS.training_density)
        size_window = arrondir(str(size_window)) 
        testing_slot = []  
        training_slots = []  
        start_index = pos_window * size_window
        end_index = start_index + size_window
        if end_index > (slot - 1):
            end_index = slot
     
        for i in range(start_index, end_index):
            testing_slot.append(i)
             
        for i in range(slot):
            if i not in (testing_slot):
                training_slots.append(i)
                      
        list_test.append(testing_slot)
        list_train.append(training_slots)
         
        if  (slot - 1) in testing_slot :
            end = True
        pos_window += 1
     
    return list_train, list_test
 
def arrondir(nombre):
    [partie_entiere, partie_decimale] = nombre.split(".")
    return int(partie_entiere) + (int(partie_decimale[0]) >= 5)

def getClusters(FLAGS):
    cluster_users = []
    cluster_services = []
    liste_cluster_users = []
    liste_cluster_services = []
    
    if FLAGS.userCluster == True:
        cluster_users, liste_cluster_users = made_kmeans_clustering(FLAGS,'user')
    if FLAGS.serviceCluster == True:
        cluster_services, liste_cluster_services = made_kmeans_clustering(FLAGS,'service')
        
    return cluster_users, cluster_services, liste_cluster_users, liste_cluster_services

def made_kmeans_clustering(FLAGS, _type):
    
    file_classes = os.path.join(os.path.dirname(__file__), '../dataset/clusters/' + str(FLAGS.qosMetric) + '_' + _type + '_classes.npy')
    classes = np.load(file_classes)
    
    list_cluster = []
    df_cluster = []
    
    ###############################
    
#     ii, = np.where(classes == 4)
#     np.put(classes, ii, 3)
#     iii, = np.where(classes == 5)
#     np.put(classes, iii, 3)
#     
#     iii, = np.where(classes == 6)
#     np.put(classes, iii, 4)
#     iii, = np.where(classes == 7)
#     np.put(classes, iii, 5)
    ###############################
    for k in range (8):
        i, = np.where(classes == k)
        if (i.size != 0):
            list_cluster.append(k)
            if _type == 'service':
                i = i.astype(str)
            df_cluster.append(i)
    
    return df_cluster, list_cluster

