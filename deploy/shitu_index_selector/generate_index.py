import cv2
import os
import random
import shutil
from gallery_builder import GalleryBuilder


def random_sample(config, method, images_list, gallery_num, output_idr):
    output_path = os.path.join(output_idr, method)
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)
    img_out_dir = os.path.join(output_path, 'gallery')
    if os.path.exists(img_out_dir) is False:
        os.mkdir(img_out_dir)
    index_out_dir = os.path.join(output_path, 'index')
    if os.path.exists(index_out_dir) is False:
        os.mkdir(index_out_dir)

    gallery_builder = GalleryBuilder(config, index_out_dir)

    gallery_images = []
    gallery_docs = []
    for class_name in images_list:
        image_save_dir = os.path.join(img_out_dir, class_name)
        if os.path.exists(image_save_dir) is False:
            os.mkdir(image_save_dir)

        imgs_num = len(images_list[class_name])
        if imgs_num < gallery_num:
            print("%s has no enough images to sample..." % (class_name))
            sample_ind = range(0, imgs_num)
        else:
            sample_ind = random.sample(range(0, imgs_num), gallery_num)

        for ind in sample_ind:
            img_file = images_list[class_name][ind]
            shutil.copy(img_file, image_save_dir)
            img_dir = os.path.join(image_save_dir, img_file.split('/')[-1])
            gallery_images.append(img_dir)
            gallery_docs.append("%s\t%s" % (img_dir, class_name))

    gallery_builder.build(config['IndexProcess'], gallery_images, gallery_docs)

    return


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
            gallery_docs.append("%s\t%s" % (img_file, class_name))

    gallery_builder.build(config['IndexProcess'], gallery_images, gallery_docs)

    return
