import tensorflow as tf
import numpy as np
import time
import sys
import logging
import os.path
 
from data_utils import wsdream_data
from train import train_deep_ae
from predict import validation_helper 
 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
 
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
 
 
FLAGS = tf.app.flags.FLAGS

 
# Hyper parameters
tf.app.flags.DEFINE_boolean('l2_reg', False, 'L2 regularization.') 
tf.app.flags.DEFINE_float('lambda_', 0.0001, 'Wight decay factor.')
tf.app.flags.DEFINE_integer('encode_act', 4, '0:linear, 1:sigmoid, 2:tanh, 3:softmax, 4:relu, 5:softplus, 6:elu, 7:selu, 8:relu6, 9:leaky_relu')
tf.app.flags.DEFINE_integer('decode_act', 4, '0:linear, 1:sigmoid, 2:tanh, 3:softmax, 4:relu, 5:softplus, 6:elu, 7:selu, 8:relu6 9:leaky_relu')
tf.app.flags.DEFINE_integer('Optimizer', 0, '0 -> GradientDescentOptimizer | 1 -> AdamOptimizer')
tf.app.flags.DEFINE_float('learning_rate', 0.05, '')
 
 
# Training parameters
tf.app.flags.DEFINE_integer('num_epoches', 1000, 'total epoch')
tf.app.flags.DEFINE_integer('num_batches', -1, 'number of batches')
tf.app.flags.DEFINE_integer('num_samples', -1, 'number of test samples')
tf.app.flags.DEFINE_integer('batch_size', 142, 'Size of the training batch.')
tf.app.flags.DEFINE_float('training_density', 0.8, 'size of training data set')
 
# Model parameters
tf.app.flags.DEFINE_boolean('userCluster', False, 'use k-means cluster for users')
tf.app.flags.DEFINE_boolean('serviceCluster', False, 'use k-means cluster for services')
tf.app.flags.DEFINE_integer('qosMetric', 0, '0 -> response time metric , 1 -> throughput metric')
tf.app.flags.DEFINE_integer('input_dim', 0, 'col count in input')
# tf.app.flags.DEFINE_integer('service_input_dim', 4500, 'col count in input')
# tf.app.flags.DEFINE_integer('user_input_dim', 142, 'col count in input')
tf.app.flags.DEFINE_string('typePrediction', 'user', 'user -> user x service | service -> service x user')
tf.app.flags.DEFINE_list('layer_sizes', [], 'size of hidden layers for user x service')
tf.app.flags.DEFINE_list('userService_layer_sizes', [1024,512,256,128], 'size of hidden layers for user x service')
tf.app.flags.DEFINE_list('serviceUser_layer_sizes', [128, 64, 32], 'size of hidden layers for service x user')
tf.app.flags.DEFINE_integer('deep', 3, 'deep of deep autoencoder')
tf.app.flags.DEFINE_float('noise', 0.0, 'fraction of noise in data set train')
tf.app.flags.DEFINE_string('noiseKind','zero','zero | gauss | salt')
 
# run parameters
tf.app.flags.DEFINE_integer('model', 1, '0 -> deep_ae | 1 -> denoising_deep_ae')
tf.app.flags.DEFINE_boolean('sparsity', False, 'True -> application of sparsity on ')
tf.app.flags.DEFINE_integer('slots', 64, 'total slots selected')
tf.app.flags.DEFINE_float('gpu', 1, 'division of GPU memory')

################################################

################################################


if FLAGS.userCluster == True and FLAGS.serviceCluster == True:
    print("Un seul type de cluster ...............")
    sys.exit()
     

if (FLAGS.userCluster == True) and (FLAGS.typePrediction == 'user'):
    print("Type de prediction user --> userCluster==False ...............")
    FLAGS.userCluster =  False
    FLAGS.serviceCluster = True
    
if FLAGS.serviceCluster == True and FLAGS.typePrediction == 'service':
    print("Type de prediction service --> serviceCluster==False ...............")
    FLAGS.serviceCluster = False
    FLAGS.userCluster = True


print('Metric:\t', FLAGS.qosMetric)
print('Predict:\t', FLAGS.typePrediction)
print('Cluster on users:\t', FLAGS.userCluster)
print('Cluster on services:\t', FLAGS.serviceCluster)

for nois in [0.0]:
    FLAGS.noise = nois
    for dens in [0.8,0.5,0.2]:
        FLAGS.training_density = dens

        print("noise:\t", FLAGS.noise)
        print("density:\t", FLAGS.training_density)
    
        ############################################################################
        ''' initialize conflit parameters '''
        ############################################################################
    
        ############################################################################
        ''' initialize data set '''
        ############################################################################
        data = wsdream_data.WS_Data(FLAGS)
        list_train, list_test = validation_helper.get_train_test_slots_lists(FLAGS)
    
        ############################################################################
        '''cross validation selected model'''
        ############################################################################
    
        text = 'RMSE\tMAE\tTIME'
        text_ = 'RMSE\tMAE\tTIME'
    
        '''initialize clusters'''
        list_cluster_users, list_cluster_services, list_users_clsuters, list_services_clusters = validation_helper.getClusters(FLAGS)
        cluster_users = []
        cluster_services = []
        
        list_clusters = []
        if FLAGS.userCluster == True:
            FLAGS.serviceCluster = False
            list_clusters = list_cluster_users
        else:
            if FLAGS.serviceCluster == True:
                list_clusters = list_cluster_services
            else:
                if FLAGS.typePrediction == 'user':
                    cluster = np.arange(4500) 
                    list_clusters.append(cluster)
                else :
                    cluster = np.arange(142) 
                    list_clusters.append(cluster)
    
        total_total = [0., 0., 0.]
        s = len(list_clusters)
        
   
        for clust in range(s):       
            clusters = list_clusters[clust]
            size = len(clusters)
            

            if size == 356 :
                FLAGS.userService_layer_sizes = [81,40,20,10]
                continue
            if size == 196:
                FLAGS.userService_layer_sizes = [45,22,11,6]#[45,22,11, 10]
                #continue
            if size == 1803:        
                FLAGS.userService_layer_sizes = [410,205,103,51]
                continue
            if size == 99:        
                FLAGS.userService_layer_sizes = [24,12,6,4]
                continue
            if size == 73:        
                FLAGS.userService_layer_sizes = [17,15,12,10]#[17,8,4,2]
                continue
            if size == 60:        
                FLAGS.userService_layer_sizes = [14,12,11,10]#[14,7,3,2]
                continue
            if size == 164:        
                FLAGS.userService_layer_sizes = [40,30,20,10]#[37,19,9,5]
                continue
            if size == 1749:        
                FLAGS.userService_layer_sizes = [400,200,100,50]
                continue

            total = [0., 0., 0.]
            print("--------------------------cluster size :------------------", clust , "-----------", size)
            trace = []
            for index, (train_slots, test_slots) in enumerate(zip(list_train, list_test)):
                start = time.time()
    
                '''initialize FLAGS parameters'''
                if FLAGS.userCluster == True:
                    cluster_users = clusters
                else :
                    if FLAGS.serviceCluster == True:
                        cluster_services = clusters
                validation_helper.init_Flags(FLAGS, train_slots, test_slots, cluster_users, cluster_services)
             
                '''initialize train and test slots'''    
                train_data, corrupt_data = data.get_training_data(train_slots, cluster_users, cluster_services)
                test_data = data.get_test_data(test_slots, cluster_users, cluster_services)
         
                ''' initilaze model'''
        
                train = train_deep_ae.Train(FLAGS, validation_helper.get_tensorbordfolder(FLAGS, index))
                reslut, trace_result = train.train(train_data, test_data, corrupt_data)
                trace.append(trace_result)
             
                t_time = time.time() - start
                total[0] += reslut[0]
                total[1] += reslut[1]
                total[2] += t_time
                txt = ('%.3f\t%.3f\t%.3f') % (reslut[0], reslut[1], t_time) 
                print(txt)
                text = text + "\n" + txt
            
                del train
          
         
            index += 1
            txt = ('%.3f\t%.3f\t%.3f\t%d') % (total[0] / index, total[1] / index, total[2], size)
            print('-----------------------\n' + txt)
            text = text + "\n" + str(clust) + "-----------------------\n" + txt
            validation_helper.writeFile(FLAGS, text, trace)
            total_total[0] += size * total[0] / index
            total_total[1] += size * total[1] / index
            total_total[2] += total[2]
        print(('%.3f\t%.3f\t%.3f') % (total_total[0] / 4500, total_total[1] / 4500, total_total[2])) 