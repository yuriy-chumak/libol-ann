(define-library (otus ann)
   (export
      ; create MxN matrix
      make-matrix ; M N

      ; set matrix elements randomly to [-1..+1]
      randomize-matrix! ; matrix

      print-matrix ; print a matrix

      read-matrix write-matrix
      T dot add mul sub sigmoid! sigmoid/ sigmoid/! at mean mabs

      bytevector->matrix
      list->matrix
   )
   (import (otus lisp)
      (otus random!)
      (owl math fp)
      (scheme inexact))
   (import (otus ffi))
(begin

   (define this (dlopen "libol-ann.so"))
   (unless this
      (print "libol-ann.so not found")
      (halt 1))

   (define mnew     (dlsym this "OL_mnew")) ; create MxN matrix
   (define mrandom! (dlsym this "OL_mrandomE")) ; set matrix elements randomly to [-1..+1]
   (define mwrite   (dlsym this "OL_mwrite")) ; write matrix to the file
   (define mread    (dlsym this "OL_mread")) ; read matrix from the file
   (define bv2f     (dlsym this "OL_bv2f"))
   (define l2f      (dlsym this "OL_l2f"))
   (define f2l      (dlsym this "OL_f2l"))
   (define mdot     (dlsym this "OL_dot"))
   (define msigmoid   (dlsym this "OL_sigmoid"))  ; важно: https://uk.wikipedia.org/wiki/Передавальна_функція_штучного_нейрона
   (define msigmoid!  (dlsym this "OL_sigmoidE"))
   (define msigmoid/  (dlsym this "OL_sigmoidD"))
   (define msigmoid/! (dlsym this "OL_sigmoidDE"))

   (define msub     (dlsym this "OL_sub"))
   (define madd     (dlsym this "OL_add"))
   (define mmul     (dlsym this "OL_mul"))
   (define mT       (dlsym this "OL_T"))
   (define mAt      (dlsym this "OL_at"))
   (define mabs     (dlsym this "OL_abs"))
   (define mmean    (dlsym this "OL_mean"))

   ; public API
   (define (make-matrix M N)
      (mnew M N))

   (define (randomize-matrix! m)
      (mrandom! m))

   (define (print-matrix M)
      (for-each (lambda (i)
            (for-each (lambda (j)
                  (display (mAt M i j))
                  (display " "))
               (iota (ref M 2) 1))
            (print))
         (iota (ref M 1) 1)))

   (define read-matrix mread)
   (define write-matrix mwrite)

   (define bytevector->matrix bv2f)
   (define list->matrix l2f)

   (define T mT)
   (define dot mdot)
   (define add madd)
   (define mul mmul)
   (define sub msub)

   (define sigmoid! msigmoid!)
   (define sigmoid/ msigmoid/)
   (define sigmoid/! msigmoid/!)
   (define at mAt)
   (define mean mmean)

))