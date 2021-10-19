#!/bin/bash
dataset_url=$1

cd dataset
rm -rf ILSVRC2012
wget -nc ${dataset_url}
tar xf ILSVRC2012_val.tar
ln -s ILSVRC2012_val ILSVRC2012
cd ILSVRC2012
ln -s val_list.txt train_list.txt
cd ../../
