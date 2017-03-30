CPP = g++
SRC_DIR = src
TARGET = lake.so
CPP_FLAGS = -m64 -O3 -Wno-unused-local-typedefs -I$(SRC_DIR) -I$(SRC_DIR)/boost_1_56_0 

all:
	if [ ! -d "$(SRC_DIR)/boost_1_56_0" ]; \
	then \
		wget http://sourceforge.net/projects/boost/files/boost/1.56.0/boost_1_56_0.zip; \
		unzip boost_1_56_0.zip; \
	fi
	
	$(CPP) $(CPP_FLAGS) -o $(TARGET) -shared -fPIC $(SRC_DIR)/main-lake.cpp

