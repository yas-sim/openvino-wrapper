#!/bin/bash

if [ "$INTEL_OPENVINO_DIR" == "" ]; then
    echo "OpenVINO environment variables are not set. Run following command to set it."
    echo "source /opt/intel/openvino/bin/setupvars.sh"
    exit 1
fi

# Copy image files and class label text data from OpenVINO installed directory
cp $INTEL_OPENVINO_DIR/deployment_tools/demo/car.png .
cp $INTEL_OPENVINO_DIR/deployment_tools/demo/car_1.bmp .
cp $INTEL_OPENVINO_DIR/deployment_tools/demo/squeezenet1.1.labels synset_words.txt
cp $INTEL_OPENVINO_DIR/deployment_tools/inference_engine/samples/python/voc_labels.txt .

# Download googlenet-v1 and mobilenet-ssd models with Model Downloader and Model Converter
python3 $INTEL_OPENVINO_DIR/deployment_tools/tools/model_downloader/downloader.py --name googlenet-v1,mobilenet-ssd
python3 $INTEL_OPENVINO_DIR/deployment_tools/tools/model_downloader/converter.py  --name googlenet-v1,mobilenet-ssd --precisions FP16

# Download models for gaze estimation demo
python3 $INTEL_OPENVINO_DIR/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-0001,head-pose-estimation-adas-0001,facial-landmarks-35-adas-0002,gaze-estimation-adas-0002

pip3 install matplotlib opencv-python numpy
