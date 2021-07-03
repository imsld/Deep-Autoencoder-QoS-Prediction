import tensorflow as tf
import tqdm as tqdm
import numpy as np

from pretreatments import io_operations as io

class WS_Data():
    
    def __init__(self, FLAGS):
        
        self.FLAGS = FLAGS
        self.data_set = []
        
        self.pbar = tqdm.tqdm(range(self.FLAGS.slots))
        for slot in self.pbar:
            data = io.read_max_density_MatrixBySlot(FLAGS.qosMetric, slot)
            data = data.fillna(0)
            self.data_set.append(data)
            self.pbar.set_description('read data (slot %i)' % slot)
    
    def get_all_data(self):
        return self.data_set
    
    def create_Dataset(self, data, batch_size):
        datatSet = tf.data.Dataset.from_tensor_slices(data)
        datatSet = datatSet.batch(batch_size)
        datatSet = datatSet.prefetch(buffer_size=1)
        return datatSet
    
    def get_matrices(self, slots, usersFilter, servicesFilter):
        all_matrices = []        
        for slot in slots:
            data = self.data_set[slot]
            
            if self.FLAGS.userCluster:
                data = data.filter(usersFilter, axis=0)
            if self.FLAGS.serviceCluster:
                data = data.filter(servicesFilter)

            matrix = data.values
            
            if (self.FLAGS.typePrediction == 'service'):
                matrix = np.transpose(matrix)
            
            if (len(all_matrices) == 0):
                all_matrices = matrix
            else:
                all_matrices = np.concatenate((all_matrices, matrix), axis=0)
                
        return all_matrices
    
    def get_training_data(self, train_slots, usersFilter, servicesFilter):
        
        all_matrices = self.get_matrices(train_slots, usersFilter, servicesFilter)
        
        train_data = self.create_Dataset(all_matrices, self.FLAGS.batch_size)
        
        corrupt = None
        
        if self.FLAGS.model == 1 :
            if self.FLAGS.noise != 0.0:
                noise_all_matrices = all_matrices.copy()                
                index = np.flatnonzero(noise_all_matrices)
                if self.FLAGS.noiseKind == 'zero':
                    indice = np.random.choice(index, replace=False, size=int(index.size * self.FLAGS.noise))
                    np.put(noise_all_matrices, indice, 0.)
                    corrupt = self.create_Dataset(noise_all_matrices, self.FLAGS.batch_size)
                else:
                    if self.FLAGS.noiseKind == 'gauss':
                        noise = tf.random_normal(shape=tf.shape(noise_all_matrices), mean=0.0, stddev=0.2, dtype=tf.float64)
                        corrupt = self.create_Dataset(tf.add(noise_all_matrices, self.FLAGS.noise * noise), self.FLAGS.batch_size)
    
        return train_data, corrupt
    
    def get_test_data(self, test_slots, usersFilter, servicesFilter):
        
        all_matrices = self.get_matrices(test_slots, usersFilter, servicesFilter)
        
        test_data = self.create_Dataset(all_matrices, self.FLAGS.batch_size)        
        
        return test_data