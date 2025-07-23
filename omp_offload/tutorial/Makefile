CC = nvc
CFLAGS = -mp=gpu -Minfo=mp,accel

all: main

main: main.o model.o
	$(CC) $(CFLAGS) -o main main.o model.o

%.o: %.c model.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o main

