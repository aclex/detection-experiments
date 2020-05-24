# Inference on NVIDIA Jetson Nano

The recommended way to get all possible performance in deep learning applications on NVIDIA Jetson family platforms seems to be running it via [DeepStream](https://developer.nvidia.com/deepstream-sdk) framework. In short, DeepStream is basically based on [GStreamer](https://gstreamer.freedesktop.org/) multimedia framework, with several important elements added specifically to run deep learning models.

## Prerequisites

Before running you need to install DeepStream SDK package and CMake to build the bounding box parsing library:
```bash
sudo apt-get install deepstream cmake
```

## Backend library

Backend library is used by `nvinfer` element for model output parsing (model output tensors are parsed and filtered to the final object detections ready to be displayed, the inference itself is done in `nvinfer` element internally). In this project the backend library is in CMake C++ project, you can build it the following way (done on the Jetson Nano device):
```bash
cd jetson/bbox_parser
mkdir build && cd build
cmake ../
make
```

## Model preparation

To prepare the trained model to run on Jetson Nano you would need to export it to ONNX format using `export.py` script (on the host where the model is trained):
```bash
python export.py --model-path output/model.pth --output model.onnx
```
Now it's ready to upload to the device.

## Configuration

Object detection tasks, among other things, can be run with `nvinfer` element. For the whole application to run (with GStreamer pipeline inside) there's a separate [`deepstream-app`](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html) utility. So the whole application structure is defined in the [configuration file](app_config.txt). One can configure the main parts of the pipeline here (e.g. add streaming to RTSP or file, please consult [`deepstream-app`](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html#page/DeepStream%2520Development%2520Guide%2Fdeepstream_app_config.3.1.html%23) reference for details), while model inference-specific settings landed in `infer_config.txt` file (you can change the name of it in `app_config.txt`).

**Be VERY careful with input image normalization settings ('net-scale-factor' and 'offsets') to make them match the normalization applied to the input images in the model training. Otherwise the model is going to produce utterly insane output values running pretty successfully. See [`nvinfer`](https://docs.nvidia.com/metropolis/deepstream/plugin-manual/index.html#page/DeepStream_Plugin_Manual%2Fdeepstream_plugin_details.02.01.html%23wwpID0E0IZ0HA) element reference for details.**

You might want to change the ONNX file of the model, file containing the label names or number of classes there. There's also backend bounding box parsing library path specified, we've built it in the above steps.

## Run
After label file and number of classes all correspond to the model, we can run the application like this (on the device):
```bash
cd jetson
deepstream-app -c app_config.txt
```

On the first run the ONNX model is going to be converted to TensorRT engine and stored. It takes about 10-15 minutes, but is not necessary for the next runs, unless the model is modified. Please also note, that the application in this configuration is run synchronized, i.e. respects the FPS rate of the source. To get the maximum possible inference performance, change `sync=0` to `sync=1` for all the sinks in `app_config.txt`.

## Inference speed

| Model | Board | Data type | Speed |
|------ | ------ | ------- | ------ |
| MobileNetV3-Small + SSDLite | Jetson Nano | fp16 | ~ 62 FPS |
| MobileNetV3-Large + SSDLite | Jetson Nano | fp16 | ~ 32 FPS |
