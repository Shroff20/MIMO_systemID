# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from bitstring import BitArray


n = np.array([0, 1, 2, 3, 4000], dtype = np.int64)



def bits(f):
    bytes = (ord(b) for b in f.read())
    for b in bytes:
        for i in range(8):
            yield (b >> i) & 1






with open("my_file.bin", "wb") as binary_file:
    
    nb = n.tobytes()
    
    nb = nb + nb
    
    binary_file.write(nb)
    
    
    print(nb)
    
    
    
data = bits(open('my_file.bin', 'r'))

data = list(data)
data = np.array(data)

data = data.reshape(int(len(data)/64), 64)

print(data)

print(str(data))


bstr='0000010111110000000000000000000000000000000000000000000000000000'
bstr='0000000000000000000000000000000000000000000000000000000000000001'

value = BitArray(bin=bstr).uint64
print(value)