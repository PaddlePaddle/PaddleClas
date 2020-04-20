#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import scipy.io
import numpy as np
import os
import sys

""".mat files data format
imagelabel.mat
jpg_name   1     2     3 ... 
label      32    12    66 ... 

setid.mat data format
jpg_name(10 records in a class)  24 6 100 65 32 ...
label                            4 ...
"""
"""
Usage: 
       python generate_flower_list.py prefix_folder mode

       python generate_flower_list.py jpg train > train_list.txt
       python generate_flower_list.py jpg valid > val_list.txt
"""
data_path = sys.argv[1]
imagelabels_path='./imagelabels.mat'
setid_path='./setid.mat'

labels = scipy.io.loadmat(imagelabels_path)
labels = np.array(labels['labels'][0])
setid = scipy.io.loadmat(setid_path)

d = {}
d['train'] = np.array(setid['trnid'][0])
d['valid'] = np.array(setid['valid'][0])
d['test']=np.array(setid['tstid'][0])

for id in d[sys.argv[2]]:
   message = str(data_path)+"/image_"+str(id).zfill(5)+".jpg "+str(labels[id-1]-1)
   print(message)
