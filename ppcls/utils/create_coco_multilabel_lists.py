# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

from ppcls.utils import logger
from ppcls.utils.logger import init_logger
from pycocotools.coco import COCO
from tqdm import tqdm

init_logger()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset_dir',
        required=True,
        help='root directory for dataset')
    parser.add_argument(
        '--image_dir',
        required=True,
        help='directory for images')
    parser.add_argument(
        '--anno_path',
        required=True,
        help='coco annotation file path')
    parser.add_argument(
        '--save_name',
        default=None,
        help='will same as anno_path if got None')
    parser.add_argument(
        '--output_dir',
        default=None,
        help='output directory, and will same as '
             'dataset_dir if got None')
    parser.add_argument(
        '--save_label_name',
        action='store_true',
        help='save label name file')

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.dataset_dir
    else:
        os.makedirs(args.dataset_dir, exist_ok=True)
    if args.save_name is None:
        args.save_name = os.path.splitext(os.path.basename(args.anno_path))[0]

    image_dir = os.path.join(args.dataset_dir, args.image_dir)
    anno_path = os.path.join(args.dataset_dir, args.anno_path)
    assert os.path.exists(image_dir) and os.path.exists(anno_path), \
        ValueError("The dataset is not Found or "
                   "the folder structure is non-conformance.")
    coco = COCO(anno_path)
    cat_id_map = {
        old_cat_id: new_cat_id
        for new_cat_id, old_cat_id in enumerate(coco.getCatIds())
    }
    num_classes = len(list(cat_id_map.keys()))

    assert 'annotations' in coco.dataset, \
        'Annotation file: {} does not contains ground truth!!!'.format(anno_path)

    save_path = os.path.join(args.dataset_dir, args.save_name + '.txt')
    logger.info("Start converting {}:".format(anno_path))
    with open(save_path, 'w') as fp:
        lines = []
        for img_id in tqdm(sorted(coco.getImgIds())):
            img_info = coco.loadImgs([img_id])[0]
            img_filename = img_info['file_name']
            img_w = img_info['width']
            img_h = img_info['height']

            img_filepath = os.path.join(image_dir, img_filename)
            if not os.path.exists(img_filepath):
                logger.warning('Illegal image file: {}, '
                               'and it will be ignored'.format(img_filepath))
                continue

            if img_w < 0 or img_h < 0:
                logger.warning(
                    'Illegal width: {} or height: {} in annotation, '
                    'and im_id: {} will be ignored'.format(img_w, img_h, img_id))
                continue

            ins_anno_ids = coco.getAnnIds(imgIds=[img_id])
            instances = coco.loadAnns(ins_anno_ids)

            label = [0] * num_classes
            for instance in instances:
                label[cat_id_map[instance['category_id']]] = 1
            lines.append(img_filename + '\t' + ','.join(map(str, label)))

        fp.write('\n'.join(lines))
        fp.close()
    logger.info("Conversion completed, save to {}:".format(save_path))

    if args.save_label_name:
        label_txt_save_name = os.path.basename(
            os.path.abspath(args.dataset_dir)) + '_labels.txt'
        label_txt_save_path = os.path.join(args.dataset_dir, label_txt_save_name)
        with open(label_txt_save_path, 'w') as fp:
            label_name_list = []
            for cat in coco.cats.values():
                label_name_list.append(cat['name'])
            fp.write('\n'.join(label_name_list))
            fp.close()
        logger.info("Save label names to {}.".format(label_txt_save_path))


if __name__ == '__main__':
    main()
