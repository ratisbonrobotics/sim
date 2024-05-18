CC = nvcc
CFLAGS = -arch=sm_52

all: vectorAdd

vectorAdd: vectorAdd.cu
	$(CC) $(CFLAGS) -o $@.out $<

clean:
	rm -f vectorAdd.out