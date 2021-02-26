unetpp pytorch implementation
==================================
60 channels hyper spectral image segmentation


install
----------------------------------
requirement:

* pytorch >= 1.4
* tqdm
* tensorboardX
* opencv

```shell script
conda create -n unetpp python=3.7
conda activate unetpp
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch -y
conda install tqdm scipy scikit-learn -y
conda install tensorboardx -c conda-forge -y
conda install opencv==4.5.0 -c conda-forge -y
```
download this repository
```shell script
git clone xxxxxx
cd unetpp_pytorch 
```
use email notification
```shell script
cp .email_setting.example ~/.email_setting
gedit ~/.email_setting
```
compile
```shell script
cd ./model/ext/deform/
python setup.py install
```
prepare dataset
```shell script
# not public now
```

usage
----------------------------------
###train and test
modify config file and run main.py to train and test.
```shell script
python main.py
```
for test only, find _**'work_phase'**_ in config file and replace **_'train-val-test'_**
 with **_'test'_**.
###show
after testing, run show.py to save result.
```shell script
python show.py
```
result are saved in _**'workdir'**_/predict, from left to right they are: 
fake rgb image, prediction, ground truth and concated image.
###show_hyper
advanced show.py with cv2 gui.
```shell script
python show_hyper.py
```
