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
3. Clone repository to your system
~~~shell
$ git clone https://github.com/yas-sim/openvino-workshop-en
~~~
4. Open a command terminal
5. Set up environment variables for OpenVINO
~~~
Linux $ source /opt/intel/openvino/bin/setupvars.sh  
~~~
~~~
Windows > call "Program Files (x86)\IntelSWTools\OpenVINO\bin\setupvars.bat"
~~~

6. Download images, class label text files, and deep-learning models using a script (`prep.sh`, or `prep.bat`)
7. Run sample programs

## Requirement
This workshop requires [Intel distribution of OpenVINO toolkit](https://software.intel.com/en-us/openvino-toolkit
).

## Contribution

## Licence

[Apache2](http://www.apache.org/licenses/LICENSE-2.0.txt)

## Author

[Yasunori Shimura](https://github.com/yassim-intel)
