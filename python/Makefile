#TODO use https://stackoverflow.com/questions/6578484/telling-gcc-directly-to-link-a-library-statically
#TODO use https://docs.python.org/3/extending/building.html#building
.PHONY: all clean mrproper rebuild

CC = gcc
LD = gcc
AR = ar
RM = rm -f

CFLAGS = -Wall -Wextra -O3 -I../include -L../bin -fPIC -I/usr/include/python3.7m
LDFLAGS = -shared

NAME = pneuralnet
OBJECTS = wrappers.o init.o

BUILD_LIB = ../bin/$(NAME).so
BUILD_OBJECTS = $(foreach object, $(OBJECTS), ../build/py/$(object))

all: $(BUILD_LIB)

clean:
	$(RM) $(BUILD_OBJECTS)

mrproper: clean
	$(RM) $(BUILD_LIB)

rebuild: mrproper all

$(BUILD_LIB): $(BUILD_OBJECTS) ../bin/libneuralnet.pic.a
	$(LD) $(LDFLAGS) -o $@ $^

../build/py/%.o: %.c ../include/neuralnet.h internal.h
	$(CC) $(CFLAGS) -o $@ -c $<
