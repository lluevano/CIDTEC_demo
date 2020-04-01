#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:38:21 2020

@author: lusantlueg
"""
#generate a database saved in directories
#./db/SUBJECT_NAME/images.jpg

#store unit vector representations for all the images in the folder

import pickle
import numpy as np
import glob
import cv2
import os
import mxnet as mx


filename='test_database.pickle'
labels = []
identity_vectors = []

sym, arg_params, aux_params = mx.model.load_checkpoint('./models/shufflev2-1.5-arcface-retina/model', 42)
model = mx.mod.Module(symbol=sym, context=[mx.cpu()], label_names = None)

database_folders = glob.glob(os.path.join('db','*'))
for subject_id,folder in enumerate(database_folders):
    images = [cv2.resize(cv2.imread(file), (112,112),interpolation=cv2.INTER_AREA) for file in glob.glob(os.path.join(folder,'*.jpg'))]
    images = np.array(images)
    im_tensor = np.zeros((len(images), 3, 112, 112))    
    for img_idx in range(len(images)):
        for i in range(3):
            im_tensor[img_idx, i, :, :] = images[img_idx,:, :, 2 - i]
    
    model.bind(data_shapes=[('data', (len(images), 3, 112,112))],force_rebind=True)
    model.set_params(arg_params,aux_params)
    data = mx.ndarray.array(im_tensor)
    db = mx.io.DataBatch(data=(data,), provide_data=[('data', data.shape)])
    model.forward(db, is_train=False)
    embeddings = model.get_outputs()[0].squeeze()
    print(folder[folder.find('/')+1:])
    print(len(embeddings))
    #print(embeddings)
    if len(images)>1:
        sum_emb = np.sum(embeddings,axis=0)
        sum_emb /= sum_emb.norm()
    else:
        sum_emb = embeddings / embeddings.norm()
        
    sum_emb = sum_emb.asnumpy()
    
    
    #model._reset_bind()
    #print(sum_emb)
    identity_vectors.append(sum_emb)
    labels.append([subject_id,folder[folder.find('/')+1:]])

print(identity_vectors)
#test_embedding = 0
    
#np.array([e.asnumpy() for e in test_embedding[:,2]])
#with open('test_database.pickle', 'rb') as test_emb_file:
#    test_embedding = pickle.load(test_emb_file)

# Step 2
with open(filename, 'wb') as database_file:
  # Step 3
  pickle.dump(np.array(identity_vectors), database_file)
  

with open('labels.pickle', 'wb') as label_file:
  # Step 3
  pickle.dump(labels, label_file)