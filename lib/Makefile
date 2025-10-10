FLAGS = -Wall -g -lm -lpthread -lX11 -I ./eigen-5.0.0
OFILES = Color.o Point.o exemple_Rosenblatt.o
PATH_SRC = .

run : build
	./main
	
build : $(OFILES) main.o 
	g++ -o main *.o $(FLAGS) 

Color.o : $(PATH_SRC)/Color.cpp $(PATH_SRC)/Color.hpp
	g++ -c $< $(FLAGS) 
	
Point.o : $(PATH_SRC)/Point.cpp $(PATH_SRC)/Point.hpp
	g++ -c $< $(FLAGS) 

exemple_Rosenblatt.o : $(PATH_SRC)/exemple_Rosenblatt.cpp $(PATH_SRC)/exemple_Rosenblatt.hpp
	g++ -c $< $(FLAGS) 

main.o : main.cpp
	g++ -c $< $(FLAGS)

# valgrind : build 
# 	valgrind --track-origins=yes ./main

clean: 
	rm -f *.o main
