import cv2


class SingleImageTask:
    def __init__(self, config, engine):
        self.image_path = config.get("image_path")
        self.engine = engine

    def run(self):
        image = cv2.imread(self.image_path)
        output = self.engine.process(image)
        print(output)
