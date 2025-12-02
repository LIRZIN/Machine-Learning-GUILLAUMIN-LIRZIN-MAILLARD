FLAGS = -Wall -g -lm -lX11 -I ./eigen-5.0.0
PATH_SRC = .

OFILES = RBF.o MLP.o LinearModel.o

run : build
	./main

build : $(OFILES) main.o 
	g++ -o main *.o $(FLAGS) 

main.o : main.cpp
	g++ -c $< $(FLAGS)

RBF.o : $(PATH_SRC)/RBF.cpp $(PATH_SRC)/RBF.hpp
	g++ -c $< $(FLAGS)

MLP.o : $(PATH_SRC)/MLP.cpp $(PATH_SRC)/MLP.hpp
	g++ -c $< $(FLAGS)

LinearModel.o : $(PATH_SRC)/LinearModel.cpp $(PATH_SRC)/LinearModel.hpp
	g++ -c $< $(FLAGS)

MLP.so : 
	g++ -std=c++17 -fPIC -shared MLP.cpp ML_c_wrapper.cpp $(FLAGS) -o MLP.so

LM.so : 
	g++ -std=c++17 -fPIC -shared LinearModel.cpp LinearModel_c_wrapper.cpp -lm -lX11 -I ./eigen-5.0.0 -o LM.so

RBF.so : 
	g++ -std=c++17 -fPIC -shared RBF.cpp RBF_c_wrapper.cpp $(FLAGS) -o RBF.so

lib.so : 
	g++ -std=c++17 -fPIC -shared MLP.cpp LinearModel.cpp RBF.cpp ML_c_wrapper.cpp $(FLAGS) -O3 -o ML.so

clean: 
	rm -f *.o *.so main mainMLP
