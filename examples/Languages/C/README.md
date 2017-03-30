# Lake Model Example using C++

This folder contains an example connecting Rhodium to a C++ model.  This example uses the C++
version of the Lake Problem developed by Riddhi Singh, Tori Ward, Jon Herman, David Hadka, and
Patrick Reed, available at https://github.com/MOEAFramework/RealWorldBenchmarks.

## Compiling

A C++ compiler is required.  We suggest using MinGW on Windows.

#### Windows

Download [boost_1_56_0.zip](http://sourceforge.net/projects/boost/files/boost/1.56.0/boost_1_56_0.zip)
and extract to the `src` folder.  After extracting, you should have a folder called
`src/boost_1_56_0`.

Next, from the `src` directory, run the following command:

```
    g++ -m64 -O3 -Iboost_1_56_0 -o ../lake.dll -shared main-lake.cpp
```

The `-m64` option compiles for 64-bit architectures, which should be used with 64-bit
Python.  Remove this option to generate a DLL compatible with 32-bit Python.

#### Linux/Unix

We have provided a Makefile to simplify compiling on Linux/Unix.  Run `make` from this directory.

## Usage

Run `python lakeModelInC.py` to run this example.