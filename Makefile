CC = nvcc
CFLAGS = -arch=sm_52

all: quickhull

quickhull: quickhull.cu
	$(CC) $(CFLAGS) -o $@.out $<

clean:
	rm -f quickhull.out