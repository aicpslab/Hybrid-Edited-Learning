import os
import csv
import time
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_constrained_optimization as tfco
import RTaylorNN as tnn


# Formulate Constraint Optimization
class ExampleProblem(tfco.ConstrainedMinimizationProblem):
    def __init__(self, y, ly, params, upper_bound):
        self._y = y
        self._ly = ly
        self._params = params
        self._upper_bound = upper_bound

    @property
    def num_constraints(self):
        return 1

    def objective(self):
        return define_loss(self._y, self._ly, self._params, self._trainable_var)

    def constraints(self):
        return self._z[0][0] - self._upper_bound


def delete_specific_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def define_loss(y, ly, params, trainable_var):
    loss1 = params['dynamics_lam'] * tf.reduce_mean(tf.square(y - ly))
    loss = loss1
    return loss

def save_files(sess, csv_path, params, weights, biases):
    for key, value in weights.items():
        np.savetxt(csv_path.replace('error', key), np.asarray(sess.run(value)), delimiter=',')
    for key, value in biases.items():
        np.savetxt(csv_path.replace('error', key), np.asarray(sess.run(value)), delimiter=',')
    save_params(params)

def save_error_to_csv(errors, count, filename):
    """
    Save error values to a CSV file.

    Arguments:
    errors -- List of error values [train_error, val_error]
    count -- Current count value
    filename -- Name of the CSV file to save the errors
    """
    # Calculate the row number
    row_number = count // 100

    # Open the CSV file in append mode
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # If file is empty, write the header
            writer.writerow(['Row', 'Train Error', 'Validation Error'])
        writer.writerow([row_number, errors[0], errors[1]])

def save_params(params):
    with open(params['model_path'].replace('ckpt', 'pkl'), 'wb') as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

def try_exp(params):
    with tf.Graph().as_default():
        x, y, ly, weights, biases = tnn.create_DeepTaylor_net(params)
        trainable_var = tf.compat.v1.trainable_variables()
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00005)
        loss = define_loss(y, ly, params, trainable_var)
        grads_and_vars = optimizer.compute_gradients(loss, var_list=trainable_var)
        train_op = optimizer.apply_gradients(grads_and_vars)

        with tf.compat.v1.Session() as sess:
            saver = tf.compat.v1.train.Saver()
            sess.run(tf.compat.v1.global_variables_initializer())
            csv_path = params['model_path'].replace('model', 'error').replace('ckpt', 'csv')
            count = 0
            best_error = 10000
            start = time.time()
            finished = 0
            saver.save(sess, params['model_path'])

            print("Loading validation data", flush=True)
            valx = np.loadtxt('%s/%s_valX.csv' % (params['subnn_path'], (params['data_name'])), delimiter=',',
                              dtype=np.float32)
            valy = np.loadtxt('%s/%s_valY.csv' % (params['subnn_path'], (params['data_name'])), delimiter=',',
                              dtype=np.float32)
            print("Validation data loaded", flush=True)

            total_length = min(len(valx), len(valy))
            valx = valx[:total_length]
            valy = valy[:total_length]

            delete_specific_file("%s/error.csv" % params['subnn_path'])

            for f in range(params['number_of_data files_for_training'] * params['num_passes_per_file']):
                if finished:
                    break
                file_num = (f % params['number_of_data files_for_training']) + 1  # 1...data_train_len

                if (params['number_of_data files_for_training'] > 1) or (f == 0):  # don't keep reloading data if always same;
                    data_train_x = np.loadtxt(('%s/%s_X%d.csv' % (params['subnn_path'], params['data_name'], file_num)),
                                              delimiter=',', dtype=np.float32)
                    data_train_ly = np.loadtxt(('%s/%s_Y%d.csv' % (params['subnn_path'], params['data_name'], file_num)),
                                              delimiter=',', dtype=np.float32)
                    total_length_train = min(len(data_train_x), len(data_train_ly),len(valx),len(valy))
                    data_train_x = data_train_x[:total_length_train]
                    data_train_ly = data_train_ly[:total_length_train]

                    total_length = data_train_x.shape[0]
                    num_batches = int(np.floor(total_length / params['batch_size']))

                ind = np.arange(total_length)
                np.random.shuffle(ind)

                data_train_x = data_train_x[ind, :]
                data_train_ly = data_train_ly[ind, :]

                for step in range(params['num_steps_per_batch'] * num_batches):
                    if params['batch_size'] < data_train_x.shape[0]:
                        offset = (step * params['batch_size']) % (total_length - params['batch_size'])
                    else:
                        offset = 0
                    batch_data_train_x = data_train_x[offset:(offset + params['batch_size']), :]
                    batch_data_train_ly = data_train_ly[offset:(offset + params['batch_size']), :]
                    batch_data_valx = valx[offset:(offset + params['batch_size']), :]
                    batch_data_valy = valy[offset:(offset + params['batch_size']), :]

                    feed_dict_train = {x: np.transpose(batch_data_train_x), ly: np.transpose(batch_data_train_ly)}
                    feed_dict_train_loss = {x: np.transpose(batch_data_train_x), ly: np.transpose(batch_data_train_ly)}
                    feed_dict_val = {x: np.transpose(batch_data_valx), ly: np.transpose(batch_data_valy)}

                    ##training
                    sess.run(train_op, feed_dict=feed_dict_train)

                    if step % params['loops for val'] == 0:
                        train_error = sess.run(loss, feed_dict=feed_dict_train_loss)
                        val_error = sess.run(loss, feed_dict=feed_dict_val)
                        #if count> 93:
                          #  print('yes')

                        Error = [train_error, val_error]

                        print(Error, flush=True)
                        print(count, flush=True)

                        if count % 100 == 0:
                            save_error_to_csv(Error, count,filename='%s/error.csv' %params['subnn_path'])

                        count = count + 1
                        save_files(sess, csv_path, params, weights, biases)

            saver.restore(sess, params['model_path'])
            save_files(sess, csv_path, params, weights, biases)


def main_exp(params):
    tf.compat.v1.set_random_seed(params['seed'])
    np.random.seed(params['seed'])
    try_exp(params)