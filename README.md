Otus Lisp ann (artificial neural network) library

You have two ways to add ann library to the Ol: manual and using package manager.

Manual:
 * do "make"
 * copy ann.scm from the otus folder to the /usr/lib/ol/otus folder.
 * copy built libol-ann.so shared library to the /usr/lib folder.

Using package manager:
 * Otus ann library already included into Otus package repository. You can find it with readme [here](https://github.com/yuriy-chumak/ol-packages);
 * In few words:
   * Download and install (copy to the /usr/bin folder) [KISS](https://raw.githubusercontent.com/kisslinux/kiss/master/kiss) executable file.
   * Clone repository (git clone https://github.com/yuriy-chumak/ol-packages $USER/.kiss/ol-packages).
   * Add ol-packages directory to the global KISS_PATH variable (export KISS_PATH=$USER/.kiss/ol-packages).
   * Install libol-ann package (kiss i libol-ann).
