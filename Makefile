ELF=main
CC=g++ -std=c++17 -Wall -g
SRC=$(shell find -name '*.cpp' | grep -v .ccls-cache | grep -v hw)
OBJ=$(SRC:.cpp=.o)
$(ELF):$(OBJ)
$(OBJ):

.PHONY:clearall clean all
all:$(ELF)
clearall:
	rm -f $(OBJ) $(ELF)
clean:
	rm -f $(OBJ)
