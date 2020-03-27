# MNIST Example Project

This project contains the project files for training a network on the MNIST dataset.


## Dependencies

The code in this repository has the following dependecies:

1. [CMake 2.8.12+](https://cmake.org/download/)
2. [OpenCV v4+](https://opencv.org/releases/)
3. [davemers0160 common code repository](https://github.com/davemers0160/Common)
4. [Dlib](http://dlib.net)

Follow the instruction for each of the dependencies according to your operating system. 

## Build

### Windows

Execute the following commands in a Windows command window:

```
mkdir build
mkdir nets
mkdir results
cd build
cmake -G "Visual Studio 15 2017 Win64" -T host=x64 ..
cmake --build . --config Release
```

Or you can use the cmake-gui and set the "source code" location to the location of the CmakeLists.txt file and the set the "build" location to the build folder. 

### Linux

Execute the following commands in a terminal window:

```
mkdir build
mkdir nets
mkdir results
cd build
cmake ..
cmake --build . --config Release -- -j4
```

Or you can use the cmake-gui and set the "source code" location to the location of the CmakeLists.txt file and the set the "build" location to the build folder. Then open a terminal window and navigate to the build folder and execute the follokwing command:

```
cmake --build . --config Release -- -j4
```

The -- -j4 tells the make to use 4 cores to build the code.  This number can be set to as many cores as you have on your PC.

## Running

To run the code you have two options.  The first is to supply individual parameters described in the table below.  For parameters that were not supplied at runtime the default values will be used.





To supply the parameters at runtime they can be called in the following manner:

```
executable -x_off=0 -fps=5.1
```

The second method (preferred) is to supply all of the parameters in a single file.  Using this method all of the input parametes must be supplied and they must be in the order outlined in the example file *cam_config.txt*

To use the file enter the following:

```
executable -cfg_file=../cam_config.txt
```

It is important to note that if the output folder does not exist the program will run, but there may not be any indication that the data is not being saved.
