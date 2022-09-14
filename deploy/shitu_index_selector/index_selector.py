import os

import cv2
import numpy as np

from paddleclas.deploy.utils import logger
from paddleclas.deploy.utils import config
from paddleclas.deploy.utils.predictor import Predictor
from paddleclas.deploy.utils.get_image_list import get_image_list
from paddleclas.deploy.python.preprocess import create_operators
from paddleclas.deploy.python.postprocess import build_postprocess
from get_images_list_from_txt import get_image_list_from_txt
from tqdm import tqdm
import shutil


class GallerySelector(Predictor):
    def __init__(self, config):
        super().__init__(config["Global"],
                         config["Global"]["rec_inference_model_dir"])
        self.preprocess_ops = create_operators(config["RecPreProcess"][
            "transform_ops"])
        self.postprocess = build_postprocess(config["RecPostProcess"])

    def predict(self, images, feature_normalize=True):
        use_onnx = self.args.get("use_onnx", False)
        if not use_onnx:
            input_names = self.predictor.get_input_names()
            input_tensor = self.predictor.get_input_handle(input_names[0])

            output_names = self.predictor.get_output_names()
            output_tensor = self.predictor.get_output_handle(output_names[0])
        else:
            input_names = self.predictor.get_inputs()[0].name
            output_names = self.predictor.get_outputs()[0].name

        if not isinstance(images, (list, )):
            images = [images]
        for idx in range(len(images)):
            for ops in self.preprocess_ops:
                images[idx] = ops(images[idx])
        image = np.array(images)

        if not use_onnx:
            input_tensor.copy_from_cpu(image)
            self.predictor.run()
            batch_output = output_tensor.copy_to_cpu()
        else:
            batch_output = self.predictor.run(
                output_names=[output_names],
                input_feed={input_names: image})[0]

        if feature_normalize:
            feas_norm = np.sqrt(
                np.sum(np.square(batch_output), axis=1, keepdims=True))
            batch_output = np.divide(batch_output, feas_norm)

        if self.postprocess is not None:
            batch_output = self.postprocess(batch_output)

        return batch_output

    def get_cos_similar_matrix(self, v1):
        num = np.dot(v1, np.array(v1).T)
        denom = np.linalg.norm(
            v1, axis=1).reshape(-1, 1) * np.linalg.norm(
                v1, axis=1)
        res = num / denom
        res[np.isneginf(res)] = 0

        return 0.5 + 0.5 * res

    def select(self, images_list, gallery_num, sim_thred):
        gallery_list = {}
        query_list = {}
        for label in tqdm(images_list):
            selected_samples_num = 0
            gallery_list[label] = []
            query_list[label] = []
            class_images = []
            for img_name in images_list[label]:
                img = cv2.imread(img_name)
                img = img[:, :, ::-1]
                class_images.append(img)
            class_fea_outputs = self.predict(class_images)
            similar_matrix = self.get_cos_similar_matrix(class_fea_outputs)

            while selected_samples_num < gallery_num and similar_matrix.sum(
            ) != 0:
                gallery_index = np.argmax(similar_matrix.sum(axis=0))
                gallery_list[label].append(images_list[label][gallery_index])
                selected_samples_num += 1

                # sample most similar query images of the selected gallery image
                query_indexes = np.where(
                    similar_matrix[gallery_index] >= sim_thred)[0]
                for query_index in query_indexes:
                    similar_matrix[query_index, :] = 0
                    similar_matrix[:, query_index] = 0

                    if query_index == gallery_index:
                        continue
                    query_list[label].append(images_list[label][query_index])

        return gallery_list, query_list


def main(config):
    gallery_selector = GallerySelector(config)

    assert 'Datasets' in config.keys(), "Datasets config not found ..."
    for dataset in config["Datasets"]:
        output_path = config["Datasets"][dataset]["output_path"]
        if os.path.exists(output_path) is False:
            os.mkdir(output_path)
        images_list = get_image_list_from_txt(
            dataset, config["Datasets"][dataset]["infer_imgs"],
            config["Datasets"][dataset]["infer_path"])
        gallery_num = config["Datasets"][dataset]["gallery_num"]
        sim_thred = config["Datasets"][dataset]["sim_thred"]
        print("Selecting gallery and query of %s" % (dataset))
        gallery_list, query_list = gallery_selector.select(
            images_list, gallery_num, sim_thred)

        gallery_num_mean = np.mean(
            [len(gallery_list[i]) for i in gallery_list])
        print("Mean of gallery num: %s" % (int(gallery_num_mean)))
        query_num_mean = np.mean([len(query_list[i]) for i in query_list])
        print("Mean of query num: %s" % (int(query_num_mean)))

        gallery_save_path = os.path.join(output_path, 'Gallery')
        if os.path.exists(gallery_save_path) is False:
            os.mkdir(gallery_save_path)
        query_save_path = os.path.join(output_path, 'Query')
        if os.path.exists(query_save_path) is False:
            os.mkdir(query_save_path)

        gallery_list_txt = open(
            os.path.join(output_path, 'gallery_list.txt'), 'w')
        query_list_txt = open(os.path.join(output_path, 'query_list.txt'), 'w')

        for label in gallery_list:
            class_save_path = os.path.join(gallery_save_path, label)
            if os.path.exists(class_save_path) is False:
                os.mkdir(class_save_path)
            for img in gallery_list[label]:
                img = img.strip()
                shutil.copy(img, class_save_path)
                img = os.path.join(
                    os.path.join('Gallery', label), img.split('/')[-1])
                gallery_list_txt.write('%s %s\n' % (img, label))

        for label in query_list:
            class_save_path = os.path.join(query_save_path, label)
            if os.path.exists(class_save_path) is False:
                os.mkdir(class_save_path)
            for img in query_list[label]:
                img = img.strip()
                shutil.copy(img, class_save_path)
                img = os.path.join(
                    os.path.join('Query', label), img.split('/')[-1])
                query_list_txt.write('%s %s\n' % (img, label))


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=True)
    main(config)
