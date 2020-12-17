import numpy as np

image_size = (256, 256)
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
filter_threshold =0.4
center_variance = 0.1
size_variance = 0.2
