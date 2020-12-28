import paddlehub as hub

image_path = ["./deploy/hubserving/ILSVRC2012_val_00006666.JPEG", ]
top_k = 5
module = hub.Module(name="clas_system")
res = module.predict(paths=image_path, top_k=top_k)
for i, image in enumerate(image_path):
    print("The returned result of {}: {}".format(image, res[i]))
