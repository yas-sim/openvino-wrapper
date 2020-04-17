@echo off

if "%INTEL_OPENVINO_DIR%" == "" goto error

rem Copy image files and class label text data from OpenVINO installed directory
copy "%INTEL_OPENVINO_DIR%\deployment_tools\demo\car.png" .
copy "%INTEL_OPENVINO_DIR%\deployment_tools\demo\car_1.bmp" .
copy "%INTEL_OPENVINO_DIR%\deployment_tools\demo\squeezenet1.1.labels" synset_words.txt
copy "%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\samples\python\voc_labels.txt" .

rem Download googlenet-v1 and mobilenet-ssd models with Model Downloader and Model Converter
python "%INTEL_OPENVINO_DIR%\deployment_tools\tools\model_downloader\downloader.py" --name googlenet-v1,mobilenet-ssd
python "%INTEL_OPENVINO_DIR%\deployment_tools\tools\model_downloader\converter.py"  --name googlenet-v1,mobilenet-ssd --precisions FP16

rem Download models for gaze estimation demo
python "%INTEL_OPENVINO_DIR%\deployment_tools\tools\model_downloader\downloader.py" --name face-detection-adas-0001,head-pose-estimation-adas-0001,facial-landmarks-35-adas-0002,gaze-estimation-adas-0002

rem Download models for object tracking demo
python "%INTEL_OPENVINO_DIR%\deployment_tools\tools\model_downloader\downloader.py" --name pedestrian-detection-adas-0002,person-reidentification-retail-0031,face-detection-adas-0001,face-reidentification-retail-0095

pip install matplotlib opencv-python numpy munkres scipy

exit /B

:error
echo OpenVINO environment variables are not set. Run following command to set it.
echo call ^"C:\Program Files (x86)\IntelSWTools\OpenVINO\bin\setupvars.bat^"
exit /B
