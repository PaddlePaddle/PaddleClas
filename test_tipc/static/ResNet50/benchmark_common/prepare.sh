#!/bin/bash

pip install -r requirements.txt
cd dataset
rm -rf ILSVRC2012
wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/ImageNet1k/ILSVRC2012_val.tar
tar xf ILSVRC2012_val.tar
ln -s ILSVRC2012_val ILSVRC2012
cd ILSVRC2012
for ((i=1; i<=4; i++));do
  cat val_list.txt >> train_list.txt
done
cd ../../
