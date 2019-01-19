from register_functions import *

from ast import literal_eval as literal
import re
from math import inf
from random import randint

import logging
from time import time

# todo: сделать VerticalShiftFunction подклассом AdditiveFunction, чтобы избежать дублирование кода
# todo: в методе find_index_of_perfection перепутаны зависимые (slaved) и влияющие (master) биты, но на р-т не влияет


def dfs(graph, start, end):
    fringe = [(start, [])]
    while fringe:
        state, path = fringe.pop()
        if path and state == end:
            yield path
            continue
        for next_state in graph[state]:
            if next_state in path:
                continue
            fringe.append((next_state, path + [next_state]))


def adjacency_matrix_to_list(matrix):
    n = len(matrix)
    result = {}
    for i in range(n):
        adjacent_vertices = []
        for j in range(n):
            if matrix[i, j] != 0:
                adjacent_vertices.append(j)
            result[i] = adjacent_vertices
    return result


def adjacency_matrix_to_list_string(matrix):
    gv_list = "digraph list{\n"
    n = len(matrix)
    for j in range(n):
        for i in range(n):
            if matrix[i, j] != 0:
                gv_list += str(i) + "->" + str(j) + ";"
                gv_list += "\n"
    gv_list += "}"
    return gv_list


def get_all_circuits_length(adj_matrix):
    graph = adjacency_matrix_to_list(adj_matrix)
    cycles = [[node] + path for node in graph for path in dfs(graph, node, node)]
    not_unic_cycles_len = [len(c) - 1 for c in cycles]
    dict_len_count = {cycle_len: not_unic_cycles_len.count(cycle_len) // cycle_len for cycle_len in not_unic_cycles_len}
    keys = list(dict_len_count.keys())
    keys.sort()
    res = []
    for key in keys:
        res.append(str(key) + ": " + str(dict_len_count[key]))
    return res


def split_state_to_blocks(state, n, r):
    blocks = []
    for i in range(n):
        blocks.append(state % (1 << r))
        state >>= r
    blocks.reverse()
    return blocks


def combine_blocks_to_state(blocks, n, r):
    result = 0
    for i in range(n):
        result ^= int(blocks[(n - 1) - i]) << r * i
    return result


def register_function_runtime_logger(func_list, n, r, act_times, measure_times):
    nr = n * r
    current_time = time()
    log_name = "logs/runtime" + "_" + str(current_time) + ".log"
    logging.basicConfig(filename=log_name, level=logging.INFO, format='%(message)s')
    np.set_printoptions(threshold=np.nan)
    init_state = split_state_to_blocks(randint(0, (1 << nr) - 1), n, r)

    logging.info("Initial state: " + str(init_state) + "\n")

    for func in func_list:
        logging.info("Function:  " + str(type(func)) + "\n")
        ts = []
        for i in range(measure_times):
            print("tic!")
            logging.info(str(i) + " measurement: ")
            t1 = time()
            next_state = func.act_k_times(init_state, act_times)
            t2 = time()
            ts.append(t2 - t1)
            logging.info("\t " + str(act_times + 1) + "th state: " + str(next_state))
            logging.info("\t " + "Execution time: " + str(t2 - t1))
        logging.info("")
        logging.info("Average time: " + str(sum(((ts[i]) for i in range(len(ts)))) / len(ts)))
        logging.info("=========")


def parse_exp(path):
    with open(path) as f:
        for line in f.readlines():
            try:
                reg = re.search(r'(\(.*\)).*exp=(\d+|inf)', line)
                yield (literal(reg.group(1)), literal(reg.group(2)))  # (params, exp)
            except ValueError:
                yield ((), inf)


def log_all_exps(n=8, r=32):
    current_time = time()
    log_name = "logs/exps_" + str(n) + "_" + str(r) + "_" + str(current_time) + ".log"
    logging.basicConfig(filename=log_name, level=logging.INFO)
    np.set_printoptions(threshold=np.nan)
    # n = r = 4
    wilandt = (n * r) ** 2 - 2 * n * r + 2
    estimate = (n ** 2) * r + n * r - 2 * n
    print("Оценка Виландта: " + str(wilandt))
    print("Оценка Кореневой: " + str(estimate))

    parameters = VerticalShiftRegisterFunction.all_sets_of_parametres(n, r)

    cnt = 0
    start = 0
    end = inf
    min_exp = estimate
    for p in parameters:
        if cnt < start:
            cnt = cnt + 1
            continue
        if cnt == end:
            break
        d, c, sh = p
        exp = inf
        f = VerticalShiftRegisterFunction(d, c, sh, n=n, r=r)
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
    log_name = "logs\iops_" + str(n) + "_" + str(r) + "_" + str(current_time) + ".log"
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
        if exp == inf:
            continue
        func = VerticalShiftRegisterFunction(d, s, sh, n=n, r=r)
        iop = find_index_of_perfection(func, exp)
        logging.info(str(p) + ";exp=" + str(exp) + ";iop=" + str(iop))
        count += 1


def binary_matrix_power(nparray, n):
    if n == 0:
        return np.identity(nparray.__len__(), dtype=np.uint64)
    if n == 1:
        return nparray
    bin_view = bin(n)[2:]
    length = len(bin_view)
    result = nparray
    for i in range(length - 1):
        result = result @ result
        result[result >= 1] = 1
    extra = n - (1 << (length - 1))
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
        for master_bit_num in range(nr):
            all_dependencies = False
            slaved_bits = 0  # ноль - младший; порядок в этом методе не важен, тщмта (но это не точно)
            print("\tmasterbit = " + str(master_bit_num))
            maximum = 1 << nr
            for state_ in range((1 << nr)):
                state_ = randint(1, maximum - 1)
                neighbor_state = split_state_to_blocks(state_ ^ (1 << master_bit_num), n, r)
                acted_state = func.act_k_times(
                    split_state_to_blocks(state_, n, r),
                    power)
                acted_neigh_state = func.act_k_times(neighbor_state, power)
                xor = combine_blocks_to_state(acted_state, n, r) ^ combine_blocks_to_state(acted_neigh_state, n, r)
                slaved_bits = slaved_bits | xor
                print("State " + bin(state_) + " has been checked")
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
