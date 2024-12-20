# -*- coding: utf-8 -*-
"""LiveClassroom-CS677-L1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Sqf2OU-eG_aC3UNLXDl3_dnvzh50-RPj

### Dynamic Typing vs Static Typing
"""

# static typing
# int x = a;
# string y = "123";

# dynamic typing
# a = 5
# b = "123"

a = 5

type(a)

b = "123"

type(b)

# static languages = error handling during compilation
# dynamic languages = error handling during runtime

"""### Interpretable VS Compiled"""

# Interpretable languages execute code line by line: slow

# Compiled languages are executed as one whole binary executable: fast

id(a)

id(b)

a = "123"
b = "123"

a == b

a is b

a = 1000
b = 1000

a == b

a is b

id(a), id(b)

a = -10
b = -10
5

a == b

a is b

id(a), id(b)

"""### Lists VS Tuples"""

# Python List is the implementation of Dynamic Array - hosts the objects of the same type
# C++: std::vector<int> ls;

ls = []

for i in range(100):
    ls.append(i)

ls = []

ls.append(1)
ls.append("123")
ls.append([1,2,3,])

ls

# ls

import numpy as np

np.array([1, 2, 3, 4])

ls.extend([1, 2, 3])

ls

ls[0]

ls = list(range(100))

ls[0:10:2] # [)

ls = [1, 2, 3]
id(ls)

# ls.append(4) O(1)
id(ls)

ls = ls + [4] # O(n)
id(ls)

ls

ls.pop() # O(N) or O(1)

ls.pop(0) # O(N) or (1)

N_ELEMENTS = 10000000
ls = list(range(N_ELEMENTS))

from timeit import default_timer

start = default_timer()
for i in range(N_ELEMENTS):
    ls.pop() # O(1)
end = default_timer()
# N_ELEMENTS*O(1)
print(f"Took {end - start} seconds...")

ls = list(range(N_ELEMENTS))

from timeit import default_timer

start = default_timer()
for i in range(N_ELEMENTS):
    ls.pop(0) #O(N)
end = default_timer()
# N_ELEMENTS*O(N)
print(f"Took {end - start} seconds...")

ls = [1, 2, 3]
ls[0] = 100

ls

ls = (1, 2, 3)
ls[0] = 100

ls[0:2]

# Mutable: object can be changed real-time, location stays the same
# Immutable: object cannot changed real-time, location will be different

# Computer Science HashMap And HashSet:
# Python: Dictionary

d = {1: "123", 2: "345"}

d

type(d)

d = {1, 2}

type(d)

# Immutable: means it can be the key in HashMap
# Mutable: means cannot

# Strings and tuples are immutable

"""### Strings"""

N_ELEMENTS = 100
s = "1"
for i in range(1, 100):
    s += str(i)
print(s)

N_ELEMENTS = 100
s = ['1']
for i in range(1, 100):
    s.append(str(i))
s = ''.join(s)
print(s)

# StringBuilder()

"""### List Comprehension"""

ls = []
for i in range(10):
    ls.append(i**2)
print(ls)

ls = [i**2 for i in range(10)]
print(ls)

"""### Functions"""

def MyFavouriteFunction(): #
    print('Hello')

def my_favourite_function(): #snake_case
    print('Hello')

# PEP-8

MyFavouriteFunction()

my_favourite_function()

def get_fibbonachi_number(n):
    if n == 1 or n == 2:
        return 1
    return get_fibbonachi_number(n - 1) + get_fibbonachi_number(n - 2)

get_fibbonachi_number(8)

def get_fibbonachi_number_memo(n, memo={}): #O(N)
    if n == 1 or n == 2:
        return 1
    if n in memo:
        return memo[n]
    memo[n] = get_fibbonachi_number(n - 1) + get_fibbonachi_number(n - 2)
    return memo[n]

# Commented out IPython magic to ensure Python compatibility.
# %%timeit
# get_fibbonachi_number(32)

# Commented out IPython magic to ensure Python compatibility.
# %%timeit
# get_fibbonachi_number_memo(32)

from random import random
import numpy as np
n = 1000
x_list = [ random() for i in range(n)]
y_list = [ random() for i in range(n)]
z_list = [ x_list[i] + y_list[i]
                           for i in range(n)]

# z_list

np.array([1, 2, 3, 4, 123])

"""### Python List VS Numpy Array"""

v1 = [5, 10]
v2 = [6, 11]

total_v = v1 + v2 #??
print(total_v)

v1 = np.array([5, 10])
v2 = np.array([6, 11])

total_v = v1 + v2 #??
print(total_v)

np.arange(0, 10, 2)

list(range(0, 10, 2))

x = np.array([1,2,3])
y = np.array([4,5,6])
z = x + y
z

y = np.array([5, 7, 9])
w = 2 + y # 2 - > np.array([2, 2, 2])
w



