#!/bin/bash

if [ $# != 1 ] ; then
echo "USAGE: $0 your_inference_lite_lib_path"
exit 1;
fi

mkdir -p  $1/demo/cxx/clas/debug/
cp  ../../ppcls/utils/imagenet1k_label_list.txt  $1/demo/cxx/clas/debug/
cp -r  ./*   $1/demo/cxx/clas/
cp ./config.txt  $1/demo/cxx/clas/debug/
cp ./imgs/tabby_cat.jpg  $1/demo/cxx/clas/debug/

echo "Prepare Done"
