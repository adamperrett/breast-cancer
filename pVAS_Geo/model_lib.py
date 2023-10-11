"""
This file contains several functions for training and testing the network.

@author: Geo
"""
import tensorflow as tf
import cnn_lib
import numpy as np
import random
import pandas as pd
import os
import math
import data_lib
from tqdm import tqdm


def test_model(model_path,
               data_path,
               data_file_path,
               view,
               results_path,
               image_type='raw',
               input_height=640,
               input_width=512
               ):
    
    """Produces predicted scores
        
    Parameters
    ----------
    model_path: string
        path of the tf graph file
    data_path: string
        path to the data folder
    data_file_path: string
        path to the data csv
    view: string
        mammographic view
    results_path: string
        path to the folder where results will be saved
    image_type: str, optional
        type of the mammographic image (raw or processed), by default 'raw'
    input_height: int, optional
        height of the input image, by default 640
    input_width: int, optional
        width of the input image,, by default 512
    """
    
    
    ####################################
    #   Define placeholders and tf graph
    ####################################
   
    #start a tensorflow session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
    print('Created the tf session')
    
    
    #create placeholders for input images and output classes
    x = tf.placeholder(tf.float32, shape=[None, input_height*input_width])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])
    
    
    #placeholders
    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
   
    phase = tf.placeholder(tf.bool, name='phase')
    
    # Build the VGG like architecture (the raw and processed architectures have 
    # slightly different graphs)
    if image_type =='raw':
        y_conv = cnn_lib.build_VGG_network_raw(x, 
                  keep_prob1,
                  keep_prob2, 
                  phase, 
                  [input_height, input_width])
    else:
        y_conv = cnn_lib.build_VGG_network_processed(x, 
                  keep_prob1,
                  keep_prob2, 
                  phase, 
                  [input_height, input_width]) 

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    ############################
    # Restore model
    ############################
    saver = tf.train.Saver()
    print('-------------------------')
    print('Loading model ' + model_path) 
    saver.restore(sess, model_path)
    print('-------------------------')

    print('Model restored ' + model_path)
       
    
    init = 0
    all_pred = []
    all_client_ids=[]
    all_sides = []
    
    print('Loading data paths ' + data_path)
    data_paths, test_labels, client_ids, sides = data_lib.read_data_paths(data_file_path, view) 

#    print('SES',data_paths, test_labels, client_ids, sides)


                                                                               

    batch_size = 2
    print('Data paths successfully loaded!')    
    n = int(math.ceil(len(test_labels)/np.float(batch_size)))
    
    print('Predicting VAS scores...')
    
    # define progress bar
    pbar = tqdm(total=len(test_labels))

    if (n>0):
        for i in range(n):
            data_batch, labels_batch, client_ids_batch, sides_batch = data_lib.read_data_batch(data_path, data_paths[batch_size*i :batch_size*(i+1)],
                                                                         test_labels[batch_size*i :batch_size*(i+1)],
                                                                         client_ids[batch_size*i :batch_size*(i+1)], 
                                                                         sides[batch_size*i :batch_size*(i+1)], 
                                                                         image_type)
            with sess.as_default():
                pred = y_conv.eval(feed_dict={x: data_batch, y_: labels_batch, keep_prob1: 1, keep_prob2:1, 'phase:0': 0})
                
                
            if (init == 0):
                init = 1
                all_pred = pred
                all_test_labels = labels_batch
                all_client_ids = client_ids_batch
                all_sides =  sides_batch
                
            else:    
                all_pred = np.concatenate((all_pred, pred),axis=0)
                all_test_labels = np.concatenate((all_test_labels, labels_batch),axis=0)
                all_client_ids = np.concatenate((all_client_ids, client_ids_batch),axis=0)
                all_sides = np.concatenate((all_sides, sides_batch),axis=0)

            # Display progress message
            pbar.update(pred.shape[0])
        
        
    else:
        print("No data was found")
    
    # Close progress bar
    pbar.close()

    # Save all the scores to csv
    df = pd.DataFrame()

    df['client_id'] = all_client_ids.reshape(-1)
    df['label'] = all_test_labels.reshape(-1)
    df['side'] = all_sides.reshape(-1)
    df['predicted_vas'] = all_pred.reshape(-1)

    csv_path = results_path + 'results_' + view + '_.csv'
    
    print('Writing results to file ' + csv_path)
    # Write the dataframe into a csv file
    df.to_csv(csv_path, index=False, header=True)