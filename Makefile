all: libol-ann.so

libol-ann.so: ann.c
	gcc $< -shared -fPIC -o $@ \
	-I../../include -I../../src \
	-Xlinker --export-dynamic \
	-O3 -g2
