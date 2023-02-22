(define-library (otus ann)
   (export
      ; create MxN matrix
      make-matrix ; M N
      at ; (ref mat i j)

      ; set matrix elements randomly to [-1..+1]
      randomize-matrix! ; matrix
      mzero!

      print-matrix ; print a matrix

      read-matrix write-matrix
      read-matrix!
      T dot add add! mul sub sigmoid! sigmoid/ sigmoid/! at mean
      mabs mclamp mclamp!

      bytevector->matrix
      list->matrix
      matrix->list
      vector->matrix
      reshape

      set-matrix-ref! ; set cell [m,n] value

      ; new experimental API
      σ
      make-input-layer
      make-dense-layer
      get-layer layers-count
      make-ann

      evaluate ; вычислить нейросеть, возвращает сложный объект со всеми промежуточными результатами (необходимый для дальнейшего обучения сети). непосредственно результат брать с помощью caar.
      backpropagate! ; обратно распространить ошибку (процесс обучения сети), изменяет нейросеть!
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
   (define mref     (dlsym this "OL_mref")) ; (ref matrix i j)

   (define mrandom! (dlsym this "OL_mrandomE")) ; set matrix elements randomly to [-1..+1]
   (define mzero!   (dlsym this "OL_zeroE")) ; set matrix elements to 0

   (define mwrite   (dlsym this "OL_mwrite")) ; write matrix to the file
   (define mread    (dlsym this "OL_mread")) ; (filename), read matrix from the file
   (define mread!   (dlsym this "OL_mreadE")) ; (matrix filename), read matrix from the file
   (define l2m      (dlsym this "OL_l2m"))
   (define v2m      (dlsym this "OL_v2m"))
   (define bv2m     (dlsym this "OL_bv2m"))
   (define f2l      (dlsym this "OL_f2l"))
   (define mreshape (dlsym this "OL_mreshape"))
   (define msetref! (dlsym this "OL_setrefE"))
   (define mdot     (dlsym this "OL_dot"))
   (define msigmoid (dlsym this "OL_sigmoid"))  ; важно: https://uk.wikipedia.org/wiki/Передавальна_функція_штучного_нейрона
   (define msigmoid!  (dlsym this "OL_sigmoidE"))
   (define msigmoid/  (dlsym this "OL_sigmoidD"))
   (define msigmoid/! (dlsym this "OL_sigmoidDE"))

   (define msub     (dlsym this "OL_sub"))
   (define madd     (dlsym this "OL_add"))
   (define maddE    (dlsym this "OL_addE"))
   (define mmul     (dlsym this "OL_mul"))
   (define mT       (dlsym this "OL_T"))
   (define mabs     (dlsym this "OL_abs"))
   (define mmean    (dlsym this "OL_mean"))
   (define mclamp   (dlsym this "OL_clamp"))
   (define mclamp!  (dlsym this "OL_clampE"))

   ; public API
   (define (make-matrix M N)
      (mnew M N))

   (define (matrix? m)
      (and (vector? m)
           (eq? (size m) 3)))

   (define (randomize-matrix! m)
      (mrandom! m))

   (define (print-matrix M)
      (for-each (lambda (i)
            (for-each (lambda (j)
                  (display (mref M i j))
                  (display " "))
               (iota (ref M 2) 1))
            (print))
         (iota (ref M 1) 1)))

   (define read-matrix mread)
   (define read-matrix! mread!)
   (define write-matrix mwrite)

   (define bytevector->matrix bv2m)
   (define list->matrix l2m)
   (define matrix->list f2l)
   (define vector->matrix v2m)

   (define reshape mreshape)

   (define set-matrix-ref! msetref!)

   (define T mT)
   (define dot mdot)
   (define add madd)
   (define add! maddE)
   (define mul mmul)
   (define sub msub)

   (define sigmoid! msigmoid!)
   (define sigmoid/ msigmoid/)
   (define sigmoid/! msigmoid/!)
   (define at mref)
   (define mean mmean)

   ; new experimental API
   (define σ 'sigmoid)

   ; создать слой входящих значений - де-факто задать размерность 
   (define (make-input-layer count)
      (list [[1 count #false] #f #f]))

   ; создать промежуточный слой из count нейронов
   ; activation-function - функция активации
   ; activation-function/ - первая производная функции активации
   ; predecessor - предыдущий слой
   (define (make-dense-layer count activation-function predecessor)
      (let*((a a/ (case activation-function
                     ('sigmoid
                        (values sigmoid! sigmoid/))
                     (else
                        (runtime-error "unknown activation function" activation-function)))))
         (define pred (car predecessor))
         (define matrix (ref pred 1))
         (cons
            [(make-matrix (ref matrix 2) count) a a/]
            predecessor)))

   ; получить n-й слой сети (начиная с 1)
   (define (get-layer ann n)
      (if (negative? n)
         (ref (lref (ref ann 2) (negate n)) 1)
         (ref (lref (reverse (ref ann 2)) n) 1)))

   (define (layers-count ann)
      (- (length (ref ann 2)) 1))

   ; создать нейросеть из топологии
   (define (make-ann layers)
      ; проинициализируем значения весов случайными числами из диапазона [0..1]
      (for-each
         (lambda (layer)
            (if (ref (ref layer 1) 3)
               (randomize-matrix! (ref layer 1))))
         layers)
      ; вернем объект "нейросеть"
      ['ann: layers #null]) ; layers values

   (define (evaluate ann data)
      ; проверка, что данные подходят по размерности
      (assert (and
         (matrix? data)
         (eq? (ref data 1) (ref (ref (last (ref ann 2) #f) 1) 1)) ; размерность выходного вектора
         (eq? (ref data 2) (ref (ref (last (ref ann 2) #f) 1) 2))
      ))
      ; наша нейросеть - это последовательный список матриц перехода между слоями нейронов (условно называемых "тензорами")
      ; это состояние постоянно меняется в процессе обучения
      ; кроме того в процессе вычисления мы получаем состояние слоев, которое можно спокойно забывать как только мы прошли один цикл обучения
      (let loop ((ann (ref ann 2)))
         (if (eq? (ref (car ann) 2) #f) ; входный слой данных
            (list (cons data #false))
         else (begin
            (define prev (loop (cdr ann))) ; предыдущий слой
            (define tensor (car ann)) ; текущая матрица слоя
            (define af (ref tensor 2)) ; функция активации

            (define layer (af (dot (caar prev) (ref tensor 1)))) ; значение следующего слоя

            (cons (cons layer tensor) prev))))
      ; теперь на выходе у нас есть список состояний слоев и тензоров между ними
      ; первый элемент - результат работы сети
      ; смысл в сохранении промежуточных значений - обучение
   )

   ; обучение сети методом обратного распространения ошибки,
   ; изменяет внутреннее состояние сети, переданной в evaluate!
   (define (backpropagate! eva error) ; backpropagate
      (assert (and
         (matrix? error)
         (eq? (ref error 1) (ref (caar eva) 1))
         (eq? (ref error 2) (ref (caar eva) 2))
      ) ===> #true)

      (let loop ((eva eva) (error error) (d2 #f) (syn #f))
         (define layer (caar eva)) ; значение слоя
         (define tensor (cdar eva)); матрица преобразования между слоями (а так же функция активации и ее производная)

         ;; (define error (if tensor
         ;;    (if answer ; ожидаемый ответ сети
         ;;       (sub answer layer)
         ;;       (dot d2 (T syn)))))
         (define delta (if tensor
            (mul
               (or error (dot d2 (T syn)))
               ((ref tensor 3) layer))))

         (if syn ; матрица предыдущего слоя - теперь ее можно учить
            (add! syn (dot (T layer) d2)))

         (if tensor
            (loop (cdr eva) #false delta (ref tensor 1)))))
))
