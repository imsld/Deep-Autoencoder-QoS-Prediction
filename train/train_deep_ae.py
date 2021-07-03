import os

from tqdm import tqdm

from model import deep_ae
import tensorflow as tf


class Train:
     
    def __init__(self, FLAGS, folder):
         
        self.FLAGS = FLAGS
        self.num_epoch = FLAGS.num_epoches
        self.num_batches = FLAGS.num_batches
        self.num_samples = FLAGS.num_samples
         
        self.min_test_rmse_loss = 1000. 
        self.min_test_mae_loss = 1000.   

        self.result = []
        self.data_trace = []
        self.min_index = -1
        
        path = '../results/tensorboard/' + folder 
        results_path = os.path.join(os.path.dirname(__file__), path)
        self.tensorboard_path = results_path
         
    def train(self, train_data, test_data, corrupt_data):
         
        self.num_epoch = self.FLAGS.num_epoches
        self.num_batches = self.FLAGS.num_batches
        self.num_samples = self.FLAGS.num_samples
                 
        self.min_test_rmse_loss = 1000.
        self.min_test_mae_loss = 1000.
         
        self.result = []
        self.data_trace = []
        self.min_index = -1
         
        Deep_ae_model = deep_ae.Deep_AE(self.FLAGS)

        iter_train = train_data.make_initializable_iterator()
        iter_test = test_data.make_initializable_iterator()
         
        train = iter_train.get_next()
        test = iter_test.get_next()
        
        corrupt = None
        if (self.FLAGS.model == 1) :
            if self.FLAGS.noise != 0.0:
                iter_corrupt = corrupt_data.make_initializable_iterator()
                corrupt = iter_corrupt.get_next()
         
        train_optimizer, train_RMSE_loss = Deep_ae_model.autoencoder_optimizer(train, corrupt)
        pred_op, test_rmse_loss_op, test_mae_loss_op = Deep_ae_model.autoencoder_validation(test) 
        
        # training_summary = tf.summary.scalar("training RMSE loss", train_RMSE_loss)
        # testing_summary = tf.summary.scalar("testing RMSE loss", test_rmse_loss_op)
         
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=self.FLAGS.gpu))) as sess:
            
            # writer = tf.summary.FileWriter(logdir=self.tensorboard_path, graph=sess.graph)
            # tf.summary.merge_all()
             
            sess.run(tf.global_variables_initializer())
             
            pbar = tqdm(range(self.num_epoch))
            for epoch in pbar:
                
                sess.run(iter_train.initializer)
                if self.FLAGS.model == 1 :
                    if self.FLAGS.noise != 0.0:
                        sess.run(iter_corrupt.initializer)
                 
                train_rmse_loss = 0
                for step in range(self.num_batches):
                    _, rmse_loss = sess.run((train_optimizer, train_RMSE_loss))
                    # _, rmse_loss, train_summ = sess.run((train_optimizer, train_RMSE_loss, training_summary))
                     
                    train_rmse_loss += rmse_loss
 
                sess.run(iter_test.initializer)
                 
                test_rmse_loss = 0
                test_mae_loss = 0
                for step in range(self.num_samples):
                    _, rmse_loss, mae_loss = sess.run((pred_op, test_rmse_loss_op, test_mae_loss_op))
                    # _, rmse_loss, mae_loss,test_summ = sess.run((pred_op, test_rmse_loss_op, test_mae_loss_op, testing_summary))
                    test_rmse_loss += rmse_loss
                    test_mae_loss += mae_loss

                train_rmse_loss /= self.num_batches
                test_rmse_loss /= self.num_samples
                test_mae_loss /= self.num_samples
                
                t = []
                t.append(train_rmse_loss)
                t.append(test_rmse_loss)
                t.append(test_mae_loss)
                self.data_trace.append(t)
                 
                if (test_rmse_loss < self.min_test_rmse_loss):
                    self.min_test_rmse_loss = test_rmse_loss
                    self.min_test_mae_loss = test_mae_loss
                    self.min_index = epoch
                
                if (test_rmse_loss>1000)or(test_rmse_loss<0)or(test_mae_loss>1000)or(test_mae_loss<0):
                    continue
                
                
                pbar.set_description('epoch: %i, train RMSE: %.3f, test RMSE: %.3f, test MAE: %.3f (best :%i, %.3f)'  
                    % (epoch, train_rmse_loss, test_rmse_loss, test_mae_loss, self.min_index, self.min_test_rmse_loss))
                # writer.add_summary(train_summ, step)
                # writer.add_summary(test_summ, step)
         
        sess.close()
        tf.reset_default_graph()
         
        self.result.append(self.min_test_rmse_loss)
        self.result.append(self.min_test_mae_loss)
         
        return self.result, self.data_trace
    
    def save_Tensorboard(self):
        return 0
