# Point Grey Stereo Camera System
## Dependencies

This system was tested with:
* ROS 2 foxy
* Ubuntu 20.04

To run this software on Ubuntu 20.04, the following must need to be done

1. Download Ubuntu 18.04 packages of FlyCapture2 SDK and PyCapture2 SDK for Linux from https://www.flir.com/products/flycapture-sdk/
2. Add Ubuntu bionic `universe` apt repos to `/etc/apt/sources.list`. (***This can be dangerous for future installs and can conflict with Ubuntu 20.04 dependences. Be sure to remove it after you are done using it***). An example of this would be
```bash
  sudo echo "deb http://us.archive.ubuntu.com/ubuntu bionic univserse" >> /etc/apt/sources.list
  sudo apt-get update
```
3. Install FlyCapture2 dependences (refer to README in zip archive)
4. Install FlyCapture2
5. Install PyCapture2 with python 3.8
```bash
  python3 setup.py install # in the PyCapture2 unzipped archive
```
