all: libol-ann.so

libol-ann.so: ann.c
	gcc $< -shared -fPIC -o $@ \
	-Xlinker --export-dynamic \
	-fopenmp