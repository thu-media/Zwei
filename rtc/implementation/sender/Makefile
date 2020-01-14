CXX=g++
CC=gcc
CPPFLAGS=-fpermissive -Wdeprecated-declarations -Wall -O3 -I/usr/local/include
CPPSOURCES=$(wildcard *.cpp)
CPPOBJECTS=$(CPPSOURCES:.cpp=.o)
CSOURCES=$(wildcard *.c)
COBJECTS=$(CSOURCES:.c=.o)
TARGET=server

.PHONY: all clean

all: .dcpp .dc $(CPPSOURCES) $(CSOURCES) $(TARGET)

.dcpp: $(CPPSOURCES)
	$(CXX) $(CPPFLAGS) -MM $(CPPSOURCES) >.dcpp
.dc: $(CSOURCES)
	$(CC) $(CFLAGS) -MM $(CSOURCES) >.dc
-include .dcpp
-include .dc
$(TARGET): $(CPPOBJECTS) $(COBJECTS)
	$(CXX) $(CPPOBJECTS) $(COBJECTS) -o $@ -L/usr/local/lib -lSDL2 -lavcodec -lavfilter -lavutil -lswresample -lswscale -lavdevice -lavformat -lpostproc -lopencv_core -lopencv_highgui -lopencv_imgproc -lSDL2_net -lx264

clean:
	rm $(CPPOBJECTS) $(COBJECTS) .dcpp .dc
install:
	sudo cp $(TARGET) /usr/local/bin
