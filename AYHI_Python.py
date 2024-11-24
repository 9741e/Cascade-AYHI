import csv
import math
import numpy as np
from numpy import random as rnd
import hashlib

hex_bin_dic = {'0':'0000', '1':'0001', '2':'0010', '3':'0011', '4':'0100', '5':'0101', '6':'0110', '7':'0111', 
               '8':'1000', '9':'1001', 'a':'1010', 'b':'1011', 'c':'1100', 'd':'1101', 'e':'1110', 'f':'1111'}

def bin_Hash(block: np.ndarray):
    block_str = ''.join([str(element) for element in block])
    hex_str = hashlib.sha1(bytes(block_str, 'utf-8')).hexdigest()
    hashed_bin_str = np.array([], dtype='int32')
    for i in hex_str:
        for j in hex_bin_dic[i]:
            hashed_bin_str = np.append(hashed_bin_str, int(j))
    return hashed_bin_str

def bin_arr_plus_one (block: np.ndarray):
    #if np.count_nonzero(block == 1) + np.count_nonzero(block == 0) != len(block):
    #    print("Ошибка: неправильный массив")
    #    exit()
    if np.count_nonzero(block == 1) == len(block):
        block[:] = np.zeros(len(block))
        print("Массив заполнен нулями")
    else:    
        count = -1
        while(True):    
             if block[count] == 1:
                 block[count] = 0
                 count -= 1
             else:
                 block[count] = 1
                 break
            
def filling(filler: np.ndarray, e_twid:np.ndarray, mask:np.ndarray):
    #if np.size(mask) == np.size(e_twid):
        #if np.count_nonzero(mask == 1) == np.size(filler):
    j = 0
    for i in range(np.size(e_twid)):
        if mask[i]:
            e_twid[i] = filler[j]
            j += 1
        #else:
            #print("Ошибка: неправильный размер наполнителя")
            #exit()
    #else:
    #    print("Ошибка: неправильный размер маски")
    #    exit()

def d(block1:np.ndarray, block2:np.ndarray):
    if np.size(block1) == np.size(block2):
        Hdistance = 0
        for i in range(np.size(block1)):
            if block1[i] != block2[i]:
                Hdistance += 1
        return Hdistance
    else:
        print("Ошибка: массивы имеют разные размеры")
        exit()
        
def looking_for_collisions(collision: int, filler_col: np.ndarray, e_twid:np.ndarray, mask:np.ndarray):
    #if np.count_nonzero(mask == 0) == np.size(filler_col):
    while np.count_nonzero(filler_col == 1) != collision:
        bin_arr_plus_one(filler_col)
    j = 0
    for i in range(np.size(e_twid)):
        if not mask[i]:
            e_twid[i] = filler_col[j]
            j += 1
    #else:
    #     print("Ошибка: неправильный размер наполнителя")
    #     exit()
    
def AYHI(BobKey: np.ndarray, e1_xor_e2: np.ndarray, Hash_r1_xor_e3: np.ndarray, collision: int, mask: np.ndarray):
    e_twid = np.copy(e1_xor_e2)
    places = np.count_nonzero(mask == 1)
    places_col = np.size(mask) - places
    filler = np.zeros(places, dtype=int)
    filler_col = np.zeros(places_col, dtype=int)
    
    if collision == 0:
        print("поиск без коллизий")
        for i in range(2**places):
            filling(filler, e_twid, mask)
            print("e_twid:", e_twid, sep='\t')
            result[expnum] += 1
            if d(Hash_r1_xor_e3, bin_Hash(BobKey[:q]^e_twid)) < (alpha + Q) / 2 * u:
                return BobKey[:q]^e_twid, True
            bin_arr_plus_one(filler)
            
    elif collision > 0:
        print(f"поиск с {collision} коллизиями")
        for i in range(2**places):
            filling(filler, e_twid, mask)
            print("e_start:", e_twid, sep='\t')
            for j in range(math.comb(places_col, collision)):
                looking_for_collisions(collision, filler_col, e_twid, mask)
                print("e_twid:", e_twid, sep='\t')
                result[expnum] += 1
                if d(Hash_r1_xor_e3, bin_Hash(BobKey[:q]^e_twid)) < (alpha + Q) / 2 * u:
                    return BobKey[:q]^e_twid, True
                bin_arr_plus_one(filler_col)
            bin_arr_plus_one(filler)
    
    return BobKey[:q], False
       
def experiment():
    n = 2*q+u # длина просеянного ключа
    
    AliceKey = rnd.randint(0, 2, size=(n))
    error_line = np.array([1 if rnd.random() < Q else 0 for i in range(n)])
    BobKey = AliceKey^error_line
    
    print(f'Ключ Алисы (первые {q} бит):', AliceKey[:q], sep='\t')
    print(f'Ключ Боба (первые {q} бит):', BobKey[:q], sep='\t')
    print('Искомый e_twid:', error_line[:q], sep='\t')
    print('e2:', error_line[q:2*q], sep='\t')

    #------------- Алиса отправляет Бобу -------------
    Hash_r1_xor_r3 = bin_Hash(AliceKey[:q])^AliceKey[2*q:]
    r1_xor_r2 = AliceKey[:q]^AliceKey[q:2*q]
    
    #------------- Боб вычисляет -------------
    r1_xor_e2 = r1_xor_r2^BobKey[q:2*q]
    e1_xor_e2 = BobKey[:q]^r1_xor_e2
    Hash_r1_xor_e3 = Hash_r1_xor_r3^BobKey[2*q:]

    print('e1_xor_e2:', e1_xor_e2, sep='\t')

    for i in range(Xc+1):
        BobKeycor, etwid_found = AYHI(BobKey, e1_xor_e2, Hash_r1_xor_e3, i, e1_xor_e2)
        if etwid_found:
            break
        
    if result[expnum] == max(result):
        BER[0] = np.count_nonzero(error_line[:q] == 1) / q
        BER[1] = np.count_nonzero(error_line[q:2*q] == 1) / q
    
    if not etwid_found:
        failure += 1
    
    if not np.array_equal(AliceKey[:q], BobKeycor):
        error += 1

    print('ключ скорректирован:', np.array_equal(AliceKey[:q], BobKeycor), sep='\t')
    

u = 160 #длина третьего блока
Xc = 4 #коллизии
alpha = 0.25 #зависит от алгоритма хешировнаия

with open('results.csv', 'w+') as f:
    writer = csv.writer(f, delimiter=';', lineterminator='\n')
    writer.writerow(['вероятность ошибки', 'длина q', 'среднее кол-во etwid', 'процент испытаний с меньшим, чем среднее кол-вом etwid', 
                     'максимальное число протестированных etwid', 'частота ошибок r1 для Nmax', 'частота ошибок r2 для Nmax', 'не найдено etwid', 'выбран неверный etwid'])

for Q in np.arange(0.005, 0.151, 0.005): # вероятность ошибки
    print("частота ошибок", round(Q, 3), sep='\t')
    
    for q in np.arange(20, 71, 10):
        print("длина q", q, sep='\t')
        
        result = np.zeros(100, dtype='int')
        BER = np.zeros(2)
        failure = 0
        error = 0

        for expnum in range(100):
            experiment()
        input('Нажмите чтобы продолжить')
        with open('results.csv', 'a') as f:
            writer = csv.writer(f, delimiter=';', lineterminator='\n')
            writer.writerow([round(Q, 3), q, round(np.mean(result), 3), np.count_nonzero(result < np.mean(result)), np.max(result), round(BER[0], 3), round(BER[1], 3), failure, error])