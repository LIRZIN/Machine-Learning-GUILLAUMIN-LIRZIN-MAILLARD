FLAGS = -Wall -g -lm -lpthread -lX11 -I ./eigen-5.0.0
PATH_SRC = .

OFILES = MultiLayerPerceptron.o

run : build
	./main

build : $(OFILES) main.o 
	g++ -o main *.o $(FLAGS) 

MultiLayerPerceptron.o : $(PATH_SRC)/MultiLayerPerceptron.cpp $(PATH_SRC)/MultiLayerPerceptron.hpp
	g++ -c $< $(FLAGS)

main.o : main.cpp
	g++ -c $< $(FLAGS)

lib.so : 
	g++ -std=c++17 -fPIC -shared MultiLayerPerceptron.cpp MultiLayerPerceptron_c_wrapper.cpp $(FLAGS) -o libMLP.so

clean: 
	rm -f *.o *.so main 
