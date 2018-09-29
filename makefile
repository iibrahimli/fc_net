TARGET = test
DEPS   = $(wildcard *.hpp)
SRC    = $(wildcard *.cpp)
OBJ    = $(SRC:.cpp=.o)
FLAGS  = -g -Wall -O3 -std=c++17

$(TARGET): $(OBJ)
	g++ $(FLAGS) $^ -o $@

%.o: %.cpp $(DEPS)
	g++ $(FLAGS) -c $< -o $@

clean: 
	rm $(TARGET) $(OBJ)
