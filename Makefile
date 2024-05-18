CC = nvcc
CFLAGS = -arch=sm_52

all: qhull

qhull: qhull.cu
	$(CC) $(CFLAGS) -o $@.out $<

clean:
	rm -f qhull.out