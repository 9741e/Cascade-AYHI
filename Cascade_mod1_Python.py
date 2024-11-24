import csv
import math
import numpy as np
from numpy import random as rnd

def parity_check(block1: np.ndarray, block2: np.ndarray):
    if np.sum(block1) % 2 != np.sum(block2) % 2:
        return True
    else:
        return False

def binary_search(Alice_block: np.ndarray, Bob_block: np.ndarray, recusion: bool, bpq_bspb_exp: np.ndarray):
    if not recusion:
        bpq_bspb_exp[0] += 1
    if(parity_check(Alice_block, Bob_block)):
        if recusion:
            bpq_bspb_exp[0] += 1
        if(np.size(Bob_block) == 1):
            Bob_block[0] = Bob_block[0]^1
        else:
            bpq_bspb_exp[1] += 1
            if(parity_check(Alice_block[0:math.floor(np.size(Alice_block)/2)], Bob_block[0:math.floor(np.size(Bob_block)/2)])):
                binary_search(Alice_block[0:math.floor(np.size(Alice_block)/2)], Bob_block[0:math.floor(np.size(Bob_block)/2)], recusion, bpq_bspb_exp)
            else:    
                binary_search(Alice_block[math.floor(np.size(Alice_block)/2):], Bob_block[math.floor(np.size(Bob_block)/2):], recusion, bpq_bspb_exp)

def shuffle_array(key: np.ndarray, row: np.ndarray):
    buffer = np.array([key[row[i]] for i in range(np.size(row))])
    for i in range(np.size(buffer)):
        key[i] = buffer[i]

def unshuffle_array(key: np.ndarray, row: np.ndarray):
    buffer = np.zeros(np.size(key))
    for i in range(np.size(buffer)):
        buffer[row[i]] = key[i]
    for i in range(np.size(buffer)):
        key[i] = buffer[i]

def Cascade(A_key: np.ndarray, B_key: np.ndarray, k: list, pass_: int, rows: np.ndarray, recursion: bool, bpq_bspb_exp: np.ndarray):
    buffer = B_key.copy()
    if pass_ != 0:
        shuffle_array(A_key, rows[pass_ - 1])
        shuffle_array(B_key, rows[pass_ - 1])
    for i in range(0, np.size(B_key), k[pass_]):
    # if np.size(b_key[i:i+k[pass_]]) == k[pass_]:# блоки фиксированной длины для данного прохода
        binary_search(A_key[i:i+k[pass_]], B_key[i:i+k[pass_]], recursion, bpq_bspb_exp)
    if pass_ != 0:
        unshuffle_array(A_key, rows[pass_ - 1])
        unshuffle_array(B_key, rows[pass_ - 1])
        if not np.array_equal(B_key, buffer):
            Cascade(A_key, B_key, k, pass_-1, rows, True, bpq_bspb_exp)

def experiment(n: int, Q: float, k: list, passes: int):
    
    bpq_bspb_exp = np.array([0, 0, 0, 0])   
    
    AliceKey = rnd.randint(0, 2, size=(n))
    BobKey = np.array([AliceKey[i]^1 if rnd.random() < Q else AliceKey[i] for i in range(n)])
    
    rows = np.array([range(n)]*(passes-1))  # инструкции перестановки

    for i in rows:
        rnd.shuffle(i)
        
    for i in range(passes):
        Cascade(AliceKey, BobKey, k, i, rows, False, bpq_bspb_exp)
    
    # выполнение алгоритма BICONF
    s=0
    while(s!=10):
        subset_indexes1 = np.array([i for i in range(n) if rnd.random()>0.5])
        AliceSet1 = AliceKey[subset_indexes1]
        BobSet1 = BobKey[subset_indexes1]
        if parity_check(AliceSet1, BobSet1):
            s = 0
            subset_indexes2 = np.array([i for i in range(n) if i not in subset_indexes1])
            AliceSet2 = AliceKey[subset_indexes2]
            BobSet2 = BobKey[subset_indexes2]
            binary_search(AliceSet1, BobSet1, False, bpq_bspb_exp)
            binary_search(AliceSet2, BobSet2, False, bpq_bspb_exp)
        else:
            s += 1
            bpq_bspb_exp[0] += 1
    
    if not np.array_equal(AliceKey, BobKey):
        bpq_bspb_exp[3] += 1

    for i in range(n):
        if AliceKey[i] != BobKey[i]:
            bpq_bspb_exp[2] += 1

    return bpq_bspb_exp

n = 10000 # длина просеянного ключа
passes = 2 # сколько проходов
k = [0]*passes # длины блоков в итерациях

with open('results.csv', 'w+') as f:
    writer = csv.writer(f, delimiter=';', lineterminator='\n')
    writer.writerow(['вероятность ошибки', 'запросов четности', 'количество битов четности при двоичном поиске', 'количество ошибок после выполнения', 
                     'минимальное колво информации для исправления', 'отказы'])

for Q in np.arange(0.005, 0.151, 0.005): # вероятность ошибки
    print(round(Q, 3))
    k[0] = math.ceil(0.92 / Q)
    k[1] = 3 * k[0]
    
    n_e = n*(-Q*math.log2(Q)-(1-Q)*math.log2(1-Q)) #среднее минимальное количество выявленной информации, 
                                                   #необходимое для исправления частоты ошибок Q
    bpq_bspb = np.array([0, 0, 0, 0])
    
    #bpq - block parity query
    #bspb - binary search parity bit
    
    for i in range(100):
        bpq_bspb += experiment(n, round(Q, 3), k, passes)
    with open('results.csv', 'a') as f:
        writer = csv.writer(f, delimiter=';', lineterminator='\n')
        writer.writerow([round(Q, 3), round(bpq_bspb[0]/100, 2), round(bpq_bspb[1]/100, 2), round(bpq_bspb[2]/100, 2), n_e, bpq_bspb[3]])
