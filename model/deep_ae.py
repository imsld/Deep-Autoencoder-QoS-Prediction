from model import _model_helper as model_helper
import tensorflow as tf


class Deep_AE:
    
    def init_var(self, layer_sizes, _input):
        
        for dim in layer_sizes:
            _name_w = 'weight_' + str(dim)
            _name_b = 'bias_' + str(dim)
            _name_bp = 'bias_prime_' + str(_input)
            
            tf.get_variable(name=_name_w, shape=(_input, dim), dtype=tf.float64, initializer=self.weight_initializer)                
            tf.get_variable(name=_name_b, shape=(dim), dtype=tf.float64, initializer=self.bias_initializer)
            tf.get_variable(name=_name_bp, shape=(_input), dtype=tf.float64, initializer=self.bias_initializer)

            _input = dim

    def __init__(self, FLAGS):
        
        self.FLAGS = FLAGS
        
        
        self.layer_sizes = FLAGS.layer_sizes
        self._input = FLAGS.input_dim
        
        self.encoding_matrices = []
              
        self.weight_initializer = model_helper._get_weight_initializer()
        self.bias_initializer = model_helper._get_bias_initializer()
        
        self.init_var(self.layer_sizes, self._input)    
                
    def encoder(self, input_x):
        next_layer_input = input_x

        for dim in self.layer_sizes:
            _name_w = 'weight_' + str(dim) + ':0'
            _name_b = 'bias_' + str(dim) + ':0'
            
            W = tf.get_default_graph().get_tensor_by_name(_name_w)            
            b = tf.get_default_graph().get_tensor_by_name(_name_b)
            
            self.encoding_matrices.append(W)
            
            output = model_helper._appli_activation(self.FLAGS, 'encode', tf.matmul(next_layer_input, W) + b)
            
#             if self.FLAGS.sparsity == True:
#                 if dim != 32:
#                     output = tf.nn.dropout(output, keep_prob=self.FLAGS.noise)
#             
            next_layer_input = output
        
        encoded_input_x = next_layer_input
        
        return encoded_input_x
    
    def decoder(self, encoded_x, input_x_shape):
        
        next_layer_input = encoded_x
        self.layer_sizes.reverse()
        self.encoding_matrices.reverse()
        
        for i, dim in enumerate(self.layer_sizes[1:] + [input_x_shape]) :
            _name_bp = 'bias_prime_' + str(dim) + ':0'
            
            W_prime = tf.transpose(self.encoding_matrices[i])          
            b_prime = tf.get_default_graph().get_tensor_by_name(_name_bp)
            
            if i != len(self.layer_sizes) - 1:
                output = model_helper._appli_activation(self.FLAGS, 'decode', tf.matmul(next_layer_input, W_prime) + b_prime)
                
            else:
                output = tf.matmul(next_layer_input, W_prime) + b_prime
            next_layer_input = output
        
        reconst_input_x = next_layer_input
        self.layer_sizes.reverse()
        self.encoding_matrices.reverse()

        return reconst_input_x   
    
    def autoencoder_inference(self, input_x):
        shape = int(input_x.get_shape()[1])
        encoded_input_x = self.encoder(input_x)
        reconst_input_x = self.decoder(encoded_input_x, shape)
        
        return reconst_input_x, encoded_input_x

    def autoencoder_optimizer(self, input_x, corrupt_data):
        data_x = input_x
        if self.FLAGS.model == 1 :
            if self.FLAGS.noise != 0.0:
                data_x = corrupt_data
        
        reconst_input_x, encoded_input_x = self.autoencoder_inference(data_x)
        
       
        
        if self.FLAGS.sparsity == True:
            RMSE_loss = (tf.reduce_mean(tf.square(data_x - reconst_input_x)))
            p = 0.01
            p_hat = tf.reduce_mean(tf.clip_by_value(encoded_input_x,1e-10,1.0),axis=0)
            kl = tf.subtract(tf.cast(p * tf.math.log(p),tf.float64), tf.cast(p * tf.math.log(p_hat), tf.float64)) 
            kl = kl +  tf.subtract(tf.cast((1 - p) * tf.math.log(1 - p),tf.float64) , tf.cast((1 - p) * tf.math.log(1 - p_hat),tf.float64))
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            RMSE_loss = RMSE_loss + self.FLAGS.lambda_ * l2_loss + tf.reduce_sum(kl)
            train_optimizer = model_helper._optimizer(RMSE_loss, self.FLAGS)
            RMSE_loss = tf.sqrt(RMSE_loss)
        else :    
            RMSE_loss = tf.sqrt(tf.reduce_mean(tf.square(data_x - reconst_input_x)))    
            train_optimizer = model_helper._optimizer(RMSE_loss, self.FLAGS)
            
        
        return train_optimizer, RMSE_loss 

    def autoencoder_validation(self, x_test):
        
        reconst_input_x, _ = self.autoencoder_inference(x_test)
        
        bool_mask, num_test_labels = model_helper.get_zero_mask(x_test)
        reconst_input_x = tf.where(bool_mask, reconst_input_x, tf.zeros_like(reconst_input_x))
        
        RMSE_loss = tf.sqrt(tf.div((tf.reduce_sum(tf.square(x_test - reconst_input_x))), num_test_labels))
        MAE_loss = tf.div(tf.reduce_sum(tf.abs(x_test - reconst_input_x)), num_test_labels)
           
        return reconst_input_x, RMSE_loss, MAE_loss