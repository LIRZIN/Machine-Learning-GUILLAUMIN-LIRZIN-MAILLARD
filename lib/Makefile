FLAGS = -Wall -g -lm -lpthread -lX11 -I ./eigen-5.0.0
PATH_SRC = .

OFILES_ROSENBLATT = Color.o Point.o exemple_Rosenblatt.o
OFILES_LM = LinearModel.o
OFILES_MLP = MultiLevelPerceptron.o

run : build
	./main

build : $(OFILES_ROSENBLATT) $(OFILES_LM) $(OFILES_MLP) main.o 
	g++ -o main *.o $(FLAGS) 
	
Rosenblatt : $(OFILES_ROSENBLATT) main.o 
	g++ -o main *.o $(FLAGS) 

LM : $(OFILES_LM) main.o 
	g++ -o main *.o $(FLAGS) 
	./main

MLP : $(OFILES_MLP) main.o 
	g++ -o main *.o $(FLAGS) 

Color.o : $(PATH_SRC)/Color.cpp $(PATH_SRC)/Color.hpp
	g++ -c $< $(FLAGS) 
	
Point.o : $(PATH_SRC)/Point.cpp $(PATH_SRC)/Point.hpp
	g++ -c $< $(FLAGS) 

exemple_Rosenblatt.o : $(PATH_SRC)/exemple_Rosenblatt.cpp $(PATH_SRC)/exemple_Rosenblatt.hpp
	g++ -c $< $(FLAGS) 

LinearModel.o : $(PATH_SRC)/LinearModel.cpp $(PATH_SRC)/LinearModel.hpp
	g++ -c $< $(FLAGS)

MultiLevelPerceptron.o : $(PATH_SRC)/MultiLevelPerceptron.cpp $(PATH_SRC)/MultiLevelPerceptron.hpp
	g++ -c $< $(FLAGS)

main.o : main.cpp
	g++ -c $< $(FLAGS)

lib.so : 
	g++ -std=c++17 -fPIC -shared MultiLevelPerceptron.cpp MultiLevelPerceptron_c_wrapper.cpp $(FLAGS) -o libMLP.so

# valgrind : build 
# 	valgrind --track-origins=yes ./main

clean: 
	rm -f *.o *.so main points.txt bg.txt MSE.txt 
