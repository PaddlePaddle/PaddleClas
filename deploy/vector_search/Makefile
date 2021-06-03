CXX=/usr/bin/g++-5

all : index

index.so : src/config.h src/graph.h src/data.h interface.cc
	$(CXX) -shared -fPIC interface.cc -o index.so -std=c++11 -Ofast  -march=native -g -flto -funroll-loops -DOMP -fopenmp
