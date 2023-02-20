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
from FRPipeline import FRP
from sklearn.preprocessing import normalize

filename='new_test_database.pickle'
labels = []
identity_vectors = []

#sym, arg_params, aux_params = mx.model.load_checkpoint('./models/shufflev2-1.5-arcface-retina/model', 42)
#model = mx.mod.Module(symbol=sym, context=[mx.cpu()], label_names = None)

fr_pipeline = FRP(False, FRP.DETECT_MOBILE_MNET, FRP.ALIGN_INSIGHTFACE, FRP.RECOGNITION_SHUFFLEFACENET)

database_folders = glob.glob(os.path.join('db','*'))
for subject_id,folder in enumerate(database_folders):
    images = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in glob.glob(os.path.join(folder,'*'))]
    #images = np.array(images)
    im_tensor = np.zeros((len(images), 112, 112, 3), dtype=np.float32)
    #for img_idx in range(len(images)):

    print(subject_id)
    print(folder)

    for i,image in enumerate(images):
        bbox, lmks = fr_pipeline.detectFaces(image)
        im_tensor[i,...] = fr_pipeline.alignCropFace(image,lmks[0])

 #   model.bind(data_shapes=[('data', (images.shape[0], 3, 112,112))], force_rebind=True)
 #   model.set_params(arg_params,aux_params)

    embeddings = fr_pipeline.batch_extract_norm_embeds(im_tensor, scale_faces=False)
    embeddings_flip = fr_pipeline.batch_extract_norm_embeds(np.flip(im_tensor,axis=2), scale_faces=False)

    #print(folder[folder.find('/')+1:])
    #print(len(embeddings))
    #print(embeddings)
    sum_emb = np.sum(np.concatenate((embeddings,embeddings_flip)), axis=0).reshape((1,embeddings.shape[-1]))
    sum_emb = sum_emb / float(embeddings.shape[0])
    #sum_emb = normalize(sum_emb)

    #model._reset_bind()
    #print(sum_emb)
    identity_vectors.append(sum_emb)
    labels.append([subject_id,folder[folder.find('/')+1:]])

#print(identity_vectors)
#test_embedding = 0
    
#np.array([e.asnumpy() for e in test_embedding[:,2]])
#with open('test_database.pickle', 'rb') as test_emb_file:
#    test_embedding = pickle.load(test_emb_file)

# Step 2
with open(filename, 'wb') as database_file:
  # Step 3
  pickle.dump({'biometric_templates': np.array(identity_vectors).reshape((len(identity_vectors),identity_vectors[0].shape[-1])), 'labels': np.array(labels)}, database_file)
  
print("done")
#with open('labels.pickle', 'wb') as label_file:
#  # Step 3
#  pickle.dump(labels, label_file)