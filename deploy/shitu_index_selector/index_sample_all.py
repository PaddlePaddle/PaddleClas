import cv2
import os
import random
import shutil
from gallery_builder import GalleryBuilder

def sample_all(config, method, images_list, output_idr):
    output_path = os.path.join(output_idr, method)
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    index_out_dir = os.path.join(output_path, 'index')
    if os.path.exists(index_out_dir) is False: 
        os.mkdir(index_out_dir)

    gallery_builder = GalleryBuilder(config, index_out_dir)

    gallery_images = []
    gallery_docs = []
    for class_name in images_list:

        imgs_num = len(images_list[class_name])
        for ind in range(imgs_num):
            img_file = images_list[class_name][ind]
            gallery_images.append(img_file)
            gallery_docs.append("%s\t%s"%(img_file, class_name))
    
    gallery_builder.build(config['IndexProcess'], gallery_images, gallery_docs)
        
    return