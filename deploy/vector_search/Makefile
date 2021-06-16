CXX=g++

ifeq ($(OS),Windows_NT)
 postfix=dll
else
 postfix=so
endif

all : index

index : src/config.h src/graph.h src/data.h interface.cc
	${CXX} -shared -fPIC interface.cc -o index.${postfix} -std=c++11 -Ofast  -march=native -g -flto -funroll-loops -DOMP -fopenmp

clean :
	rm index.${postfix}