.PHONY: all static static_pic dynamic clean mrproper rebuild

CC = gcc
LD = gcc
AR = ar
RM = rm -f

CFLAGS = -Wall -Wextra -O3 -I../include
LDFLAGS = -shared

NAME = neuralnet
OBJECTS = alloc.o activation.o io.o statistics.o tick.o getset.o
#TODO learn.o

STATIC_OBJECTS = $(foreach object, $(OBJECTS), ../build/static/$(object))
DYNAMIC_OBJECTS = $(foreach object, $(OBJECTS), ../build/pic/$(object))
STATIC_LIB = ../bin/lib$(NAME).a
STATIC_PIC_LIB = ../bin/lib$(NAME).pic.a
DYNAMIC_LIB = ../bin/lib$(NAME).so

all: static static_pic dynamic

static: $(STATIC_LIB)

static_pic: $(STATIC_PIC_LIB)

dynamic: $(DYNAMIC_LIB)

clean:
	$(RM) $(STATIC_OBJECTS) $(DYNAMIC_OBJECTS)

mrproper: clean
	$(RM) $(STATIC_LIB) $(STATIC_PIC_LIB) $(DYNAMIC_LIB)

rebuild: mrproper all

../build/static/%.o: %.c ../include/neuralnet.h internal.h
	$(CC) $(CFLAGS) -o $@ -c $<

../build/pic/%.o: %.c ../include/neuralnet.h internal.h
	$(CC) $(CFLAGS) -o $@ -c $< -fPIC

$(STATIC_LIB): $(STATIC_OBJECTS)
	$(AR) cr $@ $^

$(STATIC_PIC_LIB): $(DYNAMIC_OBJECTS)
	$(AR) cr $@ $^

$(DYNAMIC_LIB): $(DYNAMIC_OBJECTS)
	$(LD) $(LDFLAGS) -o $@ $^
