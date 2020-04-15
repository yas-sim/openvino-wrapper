## Overview
This is a tiny Python class library to wrap and abstract the OpenVINO Inference Engine. With this library, user can write a deep-leaning inferencing program easily.  
これはOpenVINOのInference EngineのラッパーライブラリでPythonで書かれています。これを使うことによってディープラーニング推論プログラムを数行で書くことが可能です。

## Description
This library conceals common initialization and processing for OpenVINO Inference Engine. User can write a few lines of code to run deep-learning inferencing with this. As the result, user will have less flexibility if they want to run an advanced inferencing with this library but the code is very short and user can easily understand and modify it if they need special features.
This library works with Intel Distribution of OpenVINO toolkit. Please make sure that you have installed and setup OpenVINO before try this.   
このライブラリはOpenVINOのInference Engineの共通の初期化処理やデータ処理をまとめてクラス化したものです。ユーザーは数行のコードを書くだけでディープラーニングの推論を行うことが可能です。結果として、複雑な処理をしようとするといろいろ制限が出ますが、ライブラリのコードは短いので自分で改造して使用することも難しくありません。  
このライブラリはIntel Distribution of OpenVINO toolkit用のライブラリです。OpenVINOをダウンロードしてセットアップをすることが必要になります。

[Intel distribution of OpenVINO toolkit](https://software.intel.com/en-us/openvino-toolkit).


## How to use
Sample programs are provided with this library. You can try them to learn how to use this library.
Before you start, you need to install and setup OpenVINO.  
サンプルプログラムが付属しています。これらを実行する前にOpenVINOのインストールとセットアップが必要です。  

1. Go to Intel distribution of OpenVINO toolkit [web page](https://software.intel.com/en-us/openvino-toolkit) and download an OpenVINO package suitable for your operating system
2. Install OpenVINO and setup support tools and accelerators (optional) by following the instruction in ['Get Started'](https://software.intel.com/en-us/openvino-toolkit/documentation/get-started) page
3. Open a command terminal
4. Clone repository to your system
~~~shell
$ git clone https://github.com/yas-sim/openvino-wrapper
~~~
5. Set up environment variables for OpenVINO
~~~
Linux $ source /opt/intel/openvino/bin/setupvars.sh  
~~~
~~~
Windows > call "Program Files (x86)\IntelSWTools\OpenVINO\bin\setupvars.bat"
~~~

6. Download images, class label text files, and deep-learning models using a script (`prep.sh`, or `prep.bat`)
7. Run sample programs

## Document

This library only supports a deep-learning model which has 1 image input and 1 output. The model with multiple inputs or outputs won't work with this library.  
This library supports both blocking (synchronous) inferencing and asynchronous inferencing.  

1. How to import this library
~~~python
import iewrap
~~~

2. API

~~~python
ieWrapper(modelFile=None, device='CPU', numRequest=4)
~~~
- *Description*
 - This function creates a `ieWrapper` object.
- *Input*
 - `modelFile`: Path to an OpenVINO IR format deep-learning model topology file (.xml). A weight file (.bin) with the same base file name wil be automatically loaded.
 - `device`: Device to run inference. E.g. `CPU`, `GPU`, `MYRIAD`, `HDDL`, `HETERO:FPGA,CPU`. Please refer to the official OpenVINO document for details.
 - `numRequest`: Maximum number of simultaneous inferencing. If you specify 4, you can run 4 inferencing task on this device at a time.  
- *Return*
 - None

~~~python
readModel(xmlFile, binFile, device='CPU', numRequest=4)
~~~
- *Description*
 - This function reads an OpenVINO IR model data. User does not need to use this function when you have read the model data in the constructor.  
- *Input*
 - `xmlFile`: Path to an OpenVINO IR format deep-learning model topology file (.xml).
 - `binFile`: Path to an OpenVINO IR format deep-learning model weight file (.xml).
 - `device`: Device to run inference. E.g. `CPU`, `GPU`, `MYRIAD`, `HDDL`, `HETERO:FPGA,CPU`. Please refer to the official OpenVINO document for details.
 - `numRequest`: Maximum number of simultaneous inferencing. If you specify 4, you can run 4 inferencing task on this device at a time.  
- *Return*
 - None

~~~python
outBlob = blockInfer(img)
~~~
- *Description*
 - Start blocking (synchronous) inferencing. The control won't back until the inference task is completed. You can immediately start processing the result after this function call. Blocking inferencing is easy to use but not efficient in terms of computer resource utilization.
- *Input*
 - `img`: OpenCV image data to infer. The image will be resized and transformed to fit to the input blob of the model. The library doesn't swap color channels (such as BGR to RGB).  
- *Return*
 - `outBlob`: Output result of the inferencing

~~~python
infID = asyncInfer(img)
~~~
- *Description*
 - Start asynchronous inferencing. Set a callback function before you call this function or the inferencing result will be wasted.
- *Input*
 - `img`: OpenCV image data to infer. The image will be resized and transformed to fit to the input blob of the model. The library doesn't swap color channels (such as BGR to RGB).  
- *Return*
 - `infID`: ID number of the requested inferencing task

~~~python
setCallback(callback)
~~~
- *Description*
 - Set a callback function which will be called after completion of each asynchronous inferencing.
- *Input*
 - `callback`: Name of the callback function. The callback function will receive 1 tuple parameter. The tuple consists of `infID` and `outBlob` `(infID, outBlob)`. You can check the inference result with `outBlob` and identify the reuslt is for which inference request by the `infID`.
- *Return*
 - None
 
## Requirement
This workshop requires [Intel distribution of OpenVINO toolkit](https://software.intel.com/en-us/openvino-toolkit
).

## Contribution

## Licence

[Apache2](http://www.apache.org/licenses/LICENSE-2.0.txt)

## Author

[Yasunori Shimura](https://github.com/yassim-intel)