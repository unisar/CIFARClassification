Steps to install and execute Tensorflow on AWS EC2 instances with GPU:

This involves installing the binary packages provided by TensorFlow and not building it from source.
When building it from source, several other things have to done. Everytime I tried doing it, I 
faced many compatibility issues. Be it with nVidia drivers or the CUDA Toolkit and Cudnn versions or Bazel.
Also, when building it from source, it takes up the disk space on g2.2xlarge and then No disk space errors are raised! 

After setting up the instance, you'll need to install gcc first and make: 

```
sudo apt-get install gcc
sudo apt-get install make
```
* Install various packages
```
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y build-essential python-pip python-dev git python-numpy swig python-dev default-jdk zip zlib1g-dev
```

* Launch g2.2xlarge instance. It's always better to add storage to the instance. Max for a free tier user is: 30G

* Blacklist Noveau which has some kind of conflict with the nvidia driver
```
echo -e "blacklist nouveau\nblacklist lbm-nouveau\noptions nouveau modeset=0\nalias nouveau off\nalias lbm-nouveau off\n" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
sudo update-initramfs -u
sudo reboot 
```

* Some other annoying thing we have to do

sudo apt-get install -y linux-image-extra-virtual
sudo reboot

* Install latest Linux headers

sudo apt-get install -y linux-source linux-headers-`` ` ``uname -r`` ` `` 

* Tensorflow binary packages for GPU version works best(rather only works) with Cuda Toolkit 7.5 and cuDNN v5.1. To install the toolkit:

```
wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run
chmod o+x cuda_7.5.18_linux.run
sudo ./cuda_7.5.18_linux.run
```

Install everything except the samples and driver. Remember to *not* to install the drivers, because tensorflow is not compatible with Nvidia Drivers version 352.39 and so the programs freeze.

To install drivers, once you are done with the above steps, do the following:

```
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/361.28/NVIDIA-Linux-x86_64-361.28.run
chmod o+x NVIDIA-Linux-x86_64-361.28.run
sudo ./NVIDIA-Linux-x86_64-361.28.run
```

* You can get cuDNN here: https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/prod/7.5/cudnn-7.5-linux-x64-v5.1-tgz
```
tar -xzf cudnn-7.5-linux-x64-v5.1.tgz 
sudo cp cuda/lib64/* /usr/local/cuda/lib64
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
```
* Export the CUDA Libraries:
```
export CUDA_HOME=/usr/local/cuda-7.5
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export PATH=${CUDA_HOME}/bin:${PATH}
```

* Install PIP: Ubuntu/Linux 64-bit

sudo apt-get install python-pip python-dev

* Ubuntu/Linux 64-bit, GPU enabled, Python 2.7. Requires CUDA toolkit 7.5 and CuDNN v5.1

export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl

* Python 2

sudo pip install --upgrade $TF_BINARY_URL

And then.. test your installation...





