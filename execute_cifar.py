
import sys
import subprocess
import multiprocessing as mp
# from Queue import Queue
from multiprocessing import Queue
from threading import Thread

def run(e, a, n, m, d, l, r):
    subprocess.call('THEANO_FLAGS=device=gpu,floatX=float32 python cifar10_cnn.py -e {0} -a {1} -n {2} -m {3} -d {4} -l {5} -r {6}'.format(e, a, n, m, d, l, r), shell=True)

epoch = [50]
aug = [False]
noise = [False]
maxout = [True, False]
dropout = [True, False]
l1 = [False]
l2 = [False]


for e in epoch:
    for a in aug:
        for n in noise:
            for m in maxout:
                for d in dropout:
                    for one in l1:
                        for two in l2:
                            run(e, a, n, m, d, one, two)
