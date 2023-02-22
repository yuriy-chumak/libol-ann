all: release

debug: CFLAGS += -O0 -g3
debug: libol-ann.so

release: CFLAGS += -O3 -g0
release: libol-ann.so

libol-ann.so: $(wildcard src/*.c)
	gcc $^ -shared -fPIC -o $@ \
	-Xlinker --export-dynamic -I. \
	$(CFLAGS) -fopenmp -lm
