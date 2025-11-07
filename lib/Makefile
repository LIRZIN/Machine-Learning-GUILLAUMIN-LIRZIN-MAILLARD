FLAGS = -Wall -g -lm -lpthread -lX11 -I ./eigen-5.0.0
PATH_SRC = .

OFILES = MLP.o

run : build
	./main

build : $(OFILES) main.o 
	g++ -o main *.o $(FLAGS) 

main.o : main.cpp
	g++ -c $< $(FLAGS)

MLP.o : $(PATH_SRC)/MLP.cpp $(PATH_SRC)/MLP.hpp
	g++ -c $< $(FLAGS)

lib.so : 
	g++ -std=c++17 -fPIC -shared -O3 MLP.cpp MLP_c_wrapper.cpp $(FLAGS) -o MLP.so

clean: 
	rm -f *.o *.so main mainMLP
