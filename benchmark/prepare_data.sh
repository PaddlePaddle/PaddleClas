#!/bin/bash
dataset_url=$1

package_check_list=(imageio tqdm Cython pycocotools tb_paddle scipy pandas wget h5py sklearn opencv-python visualdl)
for package in ${package_check_list[@]}; do
    if python -c "import ${package}" >/dev/null 2>&1; then
        echo "${package} have already installed"
    else
        echo "${package} NOT FOUND"
        pip install ${package}
        echo "${package} installed"
    fi
done

cd dataset
rm -rf ILSVRC2012
wget -nc ${dataset_url}
tar xf ILSVRC2012_val.tar
ln -s ILSVRC2012_val ILSVRC2012
cd ILSVRC2012
ln -s val_list.txt train_list.txt
cd ../../
