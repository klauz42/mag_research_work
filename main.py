import numpy as np
#from pathlib import Path
import itertools as it
from math import inf
from ast import literal_eval as literal
import re
from math import inf

import logging
from time import time

# add filemode="w" to overwrite
# logging.basicConfig(filename="index_of_perfection_256.log", level=logging.INFO)


# todo: размерность до 256
# !!!! r = 32 !!!!

def parse_exp(path):
    with open(path) as f:
        for line in f.readlines():
            try:
                reg = re.search(r'(\(.*\)).*exp=(\d+|inf)', line)
                yield (literal(reg.group(1)), literal(reg.group(2)))    #(params, exp)
            except ValueError:
                yield ((), inf)

#def cyclic_left_bit_shift(num, step, real_len=32):
#    return ((num << step) % 2**real_len) ^ (num >> (real_len - step))


def adjancy_matrix_to_list(matrix):
    gv_list = "digraph list{\n"
    n = len(matrix)
    for j in range(n):
        for i in range(n):
            if matrix[i, j] != 0:
                gv_list += str(i) + "->" + str(j) + ";"
                gv_list += "\n"
    gv_list += "}"
    return gv_list


def log_all_exps(n=8, r=32):
    current_time = time()
    log_name = "exps_" + str(n) + "_" + str(r) + "_" + str(current_time) + ".log"
    logging.basicConfig(filename=log_name, level=logging.INFO)
    np.set_printoptions(threshold=np.nan)
    # n = r = 4
    wilandt = (n * r) ** 2 - 2 * n * r + 2
    estimate = (n ** 2) * r + n * r - 2 * n
    print("Оценка Виландта: " + str(wilandt))
    print("Оценка Кореневой: " + str(estimate))

    parametres = VerticalShiftFunction.all_sets_of_parametres(n, r)

    cnt = 0
    start = 0
    end = inf
    min_exp = estimate
    for p in parametres:
        if cnt < start:
            cnt = cnt + 1
            continue
        if cnt == end:
            break
        d, c, sh = p
        exp = inf
        f = VerticalShiftFunction(d, c, sh, n=n, r=r)
        is_primitive = False
        adj_array = f.analytical_create_mixing_matrix()
        powered = adj_array
        #
        pre = binary_matrix_power(adj_array, 4096)
        if not (pre > 0).all():
            logging.info(
                str(cnt) + ") " + str(p) + "; exp=" + str(exp)
            )
            cnt = cnt + 1
            continue
        for i in range(2, min_exp + 1):
            powered = powered @ adj_array
            powered[powered > 1] = 1
            is_primitive = (powered > 0).all()

            if is_primitive:
                exp = i
                break
        logging.info(
            str(cnt) + ") " + str(p) + "; exp=" + str(exp)
            # + "\n" + str(adj_array)
        )
        cnt = cnt + 1


def log_all_iops(exp_log_file, n, r):
    current_time = time()
    log_name = "iops_" + str(n) + "_" + str(r) + "_" + str(current_time) + ".log"
    logging.basicConfig(filename=log_name, level=logging.INFO)
    count = 0
    for p, exp in parse_exp(exp_log_file):
        # if count < 7:
        #     count += 1
        #     continue
        try:
            d, s, sh = p
        except ValueError:
            count += 1
            continue
        func = VerticalShiftFunction(d, s, sh, n=n, r=r)
        iop = find_index_of_perfection(func, exp)
        logging.info(str(p) + ";exp=" + str(exp) + ";iop=" + str(iop))
        count += 1
def stupid_binary_matrix_power(nparray, n):
    result = nparray
    for i in range(n - 1):
        result = nparray @ result
        result[result > 1] = 1
    return result


def binary_matrix_power(nparray, n):
    if n == 0:
        return np.identity(nparray.__len__(), dtype=np.uint)
    if n == 1:
        return nparray
    bin_view = bin(n)[2:]
    l = len(bin_view)
    result = nparray
    for i in range(l - 1):
        result = result @ result
        result[result >= 1] = 1
    extra = n - (1 << (l - 1))
    for i in range(extra):
        result = result @ nparray
        result[result >= 1] = 1
    return result

def find_index_of_perfection(func, exp):
    n = func.n
    r = func.r
    nr = n * r
    for power in range(exp, 500):
        print("CHECKING " + str(power))
        checked_master_bits_num = 0
        need_to_try_next_power = False
        for master_bit_num in range(nr):
            all_dependencies = False
            slaved_bits = 0   # ноль - младший; порядок в этом методе не важен, тщмта (но это не точно)
            print("\tmasterbit = " + str(master_bit_num))
            for state_ in range((1 << nr)):
                neighbor_state = state_ ^ (1 << master_bit_num)
                acted_state = func.act_k_times(state_, power)
                acted_neigh_state = func.act_k_times(neighbor_state, power)
                xor = acted_state ^ acted_neigh_state
                slaved_bits = slaved_bits | xor
                # print("State " + bin(state_) + " has been checked")
                if slaved_bits == ((1 << nr) - 1):
                    all_dependencies = True
                    break
            if not all_dependencies:
                break
            else:
                checked_master_bits_num += 1
            print("\t\tChecked bits num: " + str(checked_master_bits_num))
        if checked_master_bits_num == nr:
            return power
    return inf


class Cache:

    def __init__(self, n=8, r=32):
        self.n = n
        self.r = r

    def register_filling_iterator(self):
        whole_range = (x for x in range(1 << self.r))
        all_combs_iterator = it.combinations_with_replacement(whole_range, self.n)
        return all_combs_iterator


class VerticalShiftFunction:

    def __init__(self, D, S, shift_step, n=8, r=32):
        self.D = D
        self.n = n
        self.r = r
        self.S = S
        self.shift_step = shift_step
        self.d_set_iterator = VerticalShiftFunction._generate_d_set_iterator(self.n)
        self.s_set_iterator = VerticalShiftFunction._generate_s_set_iterator(self.n)
        self.estimate = ((self.n ** 2) * self.r + self.n * self.r - 2 * self.n)

    def act(self, state):
        blocks = self.spilt_state_to_blocks(state)
        last_block = 0
        for d in self.D:
            last_block = last_block + blocks[d] # сумма
        last_block %= (1 << self.r)
        next_state = np.roll(blocks, -1, axis=0)  # сдвиг блоков регистра
        next_state[self.n - 1] = last_block
        for s in self.S:
            next_state[s] = self.cyclic_left_bit_shift(next_state[s])  # вертикальный сдвиг
        result = 0
        for i in range(self.n):
            result ^= next_state[(self.n - 1) - i] << self.r*i
        return result

    def act_k_times(self, state, k):  # k - сколько раз применить
        result = state
        for i in range(k):
            result = self.act(result)
        return result

    def spilt_state_to_blocks(self, state):
        blocks = []
        for i in range(self.n):
            blocks.append(state % (1 << self.r))
            state >>= self.r
        blocks.reverse()
        return blocks

    def cyclic_left_bit_shift(self, num):
        return ((num << self.shift_step) % (1 << self.r)) ^ (num >> (self.r - self.shift_step))

    def analytical_create_mixing_matrix(self):
        nr = self.n * self.r
        triangle_matrix = self._create_triangle_matrix()
        if self.n - 1 in self.S:
            triangle_matrix = np.roll(triangle_matrix, -self.shift_step, axis=1)
        identity_matrix = np.identity(self.r, dtype=np.uint)
        mixing_matrix = np.zeros((nr, nr), dtype=np.uint)
        for block_num in range(self.n - 1):
            if block_num not in self.S:
                mixing_matrix[(block_num + 1) * self.r: (block_num + 2) * self.r,
                block_num * self.r: (block_num + 1) * self.r] = identity_matrix
            else:
                for bit_num in range(self.r):
                    mixing_matrix[(block_num + 1) * self.r + (bit_num +
                                                              self.shift_step) % self.r, block_num * self.r + bit_num] = 1  # todo: CHECK STEP DIR
        for d in self.D:
            mixing_matrix[d * self.r: (d + 1) * self.r, (self.n - 1) * self.r: self.n * self.r] = triangle_matrix
        return mixing_matrix

    @staticmethod
    def find_all_circuits(matrix):
        pass

    @staticmethod
    def _generate_d_set_iterator(n):
        base_list = [0]

        a = [j for j in range(1, n)]
        for i in range(1, n):
            combs = tuple(it.combinations(a, i))
            for comb in combs:
                yield base_list + list(comb)

    @staticmethod
    def _generate_s_set_iterator(n):
        base_list = [n - 1]
        a = [j for j in range(n - 1)]
        for i in range(n):
            combs = tuple(it.combinations(a, i))
            for comb in combs:
                yield list(comb) + base_list

    @staticmethod
    def all_sets_of_parametres(n, r):
        d_iter = list(VerticalShiftFunction._generate_d_set_iterator(n))
        s_iter = list(VerticalShiftFunction._generate_s_set_iterator(n))
        for d in d_iter:
            for s in s_iter:
                # for step in range(1, r):
                for step in range(1, r):
                    yield (d, s, step)

    def _create_triangle_matrix(self):
        triangle_matrix = np.zeros((self.r, self.r), dtype=np.uint)
        for row_num in range(self.r):
            triangle_matrix[row_num] = \
                triangle_matrix[row_num] + \
                np.concatenate((np.ones(row_num + 1, dtype=np.uint), np.zeros(self.r - row_num - 1, dtype=np.uint)))
        return triangle_matrix

class AdditionFunction:

    def __init__(self, D, S, shift_step, n=8, r=32):
        self.D = D
        self.n = n
        self.r = r
        self.S = S
        self.shift_step = shift_step
        self.d_set_iterator = VerticalShiftFunction._generate_d_set_iterator(self.n)
        self.s_set_iterator = VerticalShiftFunction._generate_s_set_iterator(self.n)
        self.estimate = ((self.n ** 2) * self.r + self.n * self.r - 2 * self.n)

    def act(self, state):
        blocks = self.spilt_state_to_blocks(state)
        last_block = 0
        for d in self.D:
            last_block = last_block + blocks[d] # сумма
        last_block %= (1 << self.r)
        next_state = np.roll(blocks, -1, axis=0)  # сдвиг блоков регистра
        next_state[self.n - 1] = last_block
        for s in self.S:
            next_state[s] = self.cyclic_left_bit_shift(next_state[s])  # вертикальный сдвиг
        result = 0
        for i in range(self.n):
            result ^= next_state[(self.n - 1) - i] << self.r*i
        return result

    def act_k_times(self, state, k):  # k - сколько раз применить
        result = state
        for i in range(k):
            result = self.act(result)
        return result

    def spilt_state_to_blocks(self, state):
        blocks = []
        for i in range(self.n):
            blocks.append(state % (1 << self.r))
            state >>= self.r
        blocks.reverse()
        return blocks

    def cyclic_left_bit_shift(self, num):
        return ((num << self.shift_step) % (1 << self.r)) ^ (num >> (self.r - self.shift_step))

    def analytical_create_mixing_matrix(self):
        nr = self.n * self.r
        triangle_matrix = self._create_triangle_matrix()
        if self.n - 1 in self.S:
            triangle_matrix = np.roll(triangle_matrix, -self.shift_step, axis=1)
        identity_matrix = np.identity(self.r, dtype=np.uint)
        mixing_matrix = np.zeros((nr, nr), dtype=np.uint)
        for block_num in range(self.n - 1):
            if block_num not in self.S:
                mixing_matrix[(block_num + 1) * self.r: (block_num + 2) * self.r,
                block_num * self.r: (block_num + 1) * self.r] = identity_matrix
            else:
                for bit_num in range(self.r):
                    mixing_matrix[(block_num + 1) * self.r + (bit_num +
                                                              self.shift_step) % self.r, block_num * self.r + bit_num] = 1  # todo: CHECK STEP DIR
        for d in self.D:
            mixing_matrix[d * self.r: (d + 1) * self.r, (self.n - 1) * self.r: self.n * self.r] = triangle_matrix
        return mixing_matrix

    @staticmethod
    def find_all_circuits(matrix):
        pass

    @staticmethod
    def _generate_d_set_iterator(n):
        base_list = [0]

        a = [j for j in range(1, n)]
        for i in range(1, n):
            combs = tuple(it.combinations(a, i))
            for comb in combs:
                yield base_list + list(comb)

    @staticmethod
    def _generate_s_set_iterator(n):
        base_list = [n - 1]
        a = [j for j in range(n - 1)]
        for i in range(n):
            combs = tuple(it.combinations(a, i))
            for comb in combs:
                yield list(comb) + base_list

    @staticmethod
    def all_sets_of_parametres(n, r):
        d_iter = list(VerticalShiftFunction._generate_d_set_iterator(n))
        s_iter = list(VerticalShiftFunction._generate_s_set_iterator(n))
        for d in d_iter:
            for s in s_iter:
                # for step in range(1, r):
                for step in range(1, r):
                    yield (d, s, step)

    def _create_triangle_matrix(self):
        triangle_matrix = np.zeros((self.r, self.r), dtype=np.uint)
        for row_num in range(self.r):
            triangle_matrix[row_num] = \
                triangle_matrix[row_num] + \
                np.concatenate((np.ones(row_num + 1, dtype=np.uint), np.zeros(self.r - row_num - 1, dtype=np.uint)))
        return triangle_matrix

if __name__ == '__main__':

    # log_all_exps(4,4)
    # exps = 'exps.log'
    # log_all_iops(exps, 8, 32)
    p = ([0, 1, 2, 3, 4, 5, 6, 7], [0, 7], 1)
    d, s, sh = p
    f = VerticalShiftFunction(d, s, sh)
    find_index_of_perfection(f, 9)

    #
    #
    # for params, exp in parse_exp(exps):
    #     d, c, sh = p
    #     if i[]
    #     f = VerticalShiftFunction([0, 1, 2, 3], [3], 1, n=4, r=4)
    #     ind = find_index_of_perfection(f, 5)
    #     print(ind)
    #     logging.info(
    #         str(cnt) + ") " + str(p) + "; exp=" + str(exp)
    #         + "\n-----------"
    #     )
