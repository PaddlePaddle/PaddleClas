Global:
  infer_imgs: "./dataset/iCartoonFace/val2/0000000.jpg"
  det_inference_model_dir: "./output/det"
  rec_inference_model_dir: "./output/"
  batch_size: 1
  image_shape: [3, 640, 640]
  threshold: 0.5
  max_det_results: 1
  labe_list:
  - foreground

  # inference engine config
  use_gpu: True
  enable_mkldnn: True
  cpu_num_threads: 100
  enable_benchmark: True
  use_fp16: False
  ir_optim: True
  use_tensorrt: False
  gpu_mem: 8000
  enable_profile: False

DetPreProcess:
  transform_ops:
    - DetResize:
        interp: 2
        keep_ratio: false
        target_size: [640, 640]
    - DetNormalizeImage:
        is_scale: true
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
    - DetPermute: {}

DetPostProcess: {}


RecPreProcess:
  transform_ops:
    - ResizeImage:
        resize_short: 256
    - CropImage:
        size: 224
    - NormalizeImage:
        scale: 0.00392157
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
        order: ''
    - ToCHWImage:

RecPostProcess: null

# indexing engine config
IndexProcess:
  build:  
    enable: False
    index_path: "./icartoon_index/"
    image_root: "./dataset/iCartoonFace"
    data_file:  "./dataset/iCartoonFace/gallery_pesudo.txt"
    spacer: "\t"
    dist_type: "IP"
    pq_size: 100
    embedding_size: 2048
  infer:
    index_path: "./icartoon_index/"
    search_budget: 100
    return_k: 10
