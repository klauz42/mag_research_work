import numpy as np
import itertools as it


class AdditiveRegisterFunction:

    def __init__(self, D, n, r):
        self.D = D
        self.n = n
        self.r = r
        self.d_set_iterator = self._generate_d_set_iterator()
        self.estimate = ((self.n ** 2) * self.r + self.n * self.r - 2 * self.n)

    def act(self, state):
        last_block = 0
        for d in self.D:
            last_block = last_block + state[d]  # сумма
        last_block %= (1 << self.r)
        next_state = np.roll(state, -1, axis=0)  # сдвиг блоков регистра
        next_state[self.n - 1] = last_block
        return next_state

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

    def analytical_create_mixing_matrix(self):
        nr = self.n * self.r
        triangle_matrix = self._create_triangle_matrix()
        identity_matrix = np.identity(self.r*(self.n-1), dtype=np.uint64)
        mixing_matrix = np.zeros((nr, nr), dtype=np.uint64)
        mixing_matrix[self.r: self.n * self.r, self.r: self.n * self.r] = identity_matrix
        for d in self.D:
            mixing_matrix[d * self.r: (d + 1) * self.r, (self.n - 1) * self.r: self.n * self.r] = triangle_matrix
        return mixing_matrix

    def create_mixing_matrix(self):
        nr = self.n * self.r
        mixing_matrix = np.zeros((nr, nr), dtype=np.uint64)
        for slaved_bit_num in range(nr):
            master_bits = 0  # ноль - младший
            for state_ in range((1 << nr)):
                neighbor_state = state_ ^ (1 << slaved_bit_num)
                acted_state = self.act(self.spilt_state_to_blocks(state_))
                acted_neigh_state = self.act(self.spilt_state_to_blocks(neighbor_state))
                xor = self.combine_blocks_to_state(acted_state) ^ self.combine_blocks_to_state(acted_neigh_state)
                master_bits = master_bits | xor
            mixing_matrix[nr - 1 - slaved_bit_num, 0:nr] = self.int_to_bin_np_row(master_bits)
        return mixing_matrix

    def int_to_bin_np_row(self, uint):
        uint_len = bin(uint).__len__() - 2
        nr = self.n * self.r
        bin_row = np.zeros(nr, dtype=np.uint64)    # младшие биты сверху
        for i in range(uint_len):
            bin_row[nr - 1 - i] = uint % 2
            uint = uint >> 1
        return bin_row

    def combine_blocks_to_state(self, blocks):
        result = 0
        for i in range(self.n):
            result ^= int(blocks[(self.n - 1) - i]) << self.r * i
        return result

    def _generate_d_set_iterator(self):
        base_list = [0]

        a = [j for j in range(1, self.n)]
        for i in range(1, self.n):
            combs = tuple(it.combinations(a, i))
            for comb in combs:
                yield base_list + list(comb)

    def all_sets_of_parametres(self):
        d_iter = list(self._generate_d_set_iterator())
        for d in d_iter:
            for step in range(1, self.r):
                yield (d, step)

    def _create_triangle_matrix(self):
        triangle_matrix = np.zeros((self.r, self.r), dtype=np.uint64)
        for row_num in range(self.r):
            triangle_matrix[row_num] = \
                triangle_matrix[row_num] + \
                np.concatenate((np.ones(row_num + 1, dtype=np.uint64), np.zeros(self.r - row_num - 1, dtype=np.uint64)))
        return triangle_matrix


class VerticalShiftRegisterFunction(AdditiveRegisterFunction):

    def __init__(self, D, S, shift_step, n, r,):
        super().__init__(D, n, r)
        self.S = S
        self.shift_step = shift_step
        self.s_set_iterator = self._generate_s_set_iterator()
        self.estimate = ((self.n ** 2) * self.r + self.n * self.r - 2 * self.n)
        self.rbit = 1 << self.r

    def act(self, state: list):
        last_block: int = 0
        for d in self.D:
            last_block = last_block + state[d] # сумма
        last_block %= self.rbit
        next_state: list = state[1:]  # сдвиг блоков регистра
        next_state.append(last_block)
        for s in self.S:
            next_state[s] = self.cyclic_left_bit_shift(next_state[s])  # вертикальный сдвиг
        return next_state

    def cyclic_left_bit_shift(self, num):
        return ((num << self.shift_step) % self.rbit) ^ (num >> (self.r - self.shift_step))

    def analytical_create_mixing_matrix(self):
        nr = self.n * self.r
        triangle_matrix = self._create_triangle_matrix()
        if self.n - 1 in self.S:
            triangle_matrix = np.roll(triangle_matrix, -self.shift_step, axis=1)
        identity_matrix = np.identity(self.r, dtype=np.uint64)
        mixing_matrix = np.zeros((nr, nr), dtype=np.uint64)
        for block_num in range(self.n - 1):
            if block_num not in self.S:
                mixing_matrix[(block_num + 1) * self.r: (block_num + 2) * self.r,
                block_num * self.r: (block_num + 1) * self.r] = identity_matrix
            else:
                for bit_num in range(self.r):
                    mixing_matrix[(block_num + 1) * self.r + (bit_num +
                                                        self.shift_step) % self.r, block_num * self.r + bit_num] = 1
        for d in self.D:
            mixing_matrix[d * self.r: (d + 1) * self.r, (self.n - 1) * self.r: self.n * self.r] = triangle_matrix
        return mixing_matrix

    def _generate_s_set_iterator(self):
        base_list = [self.n - 1]
        a = [j for j in range(self.n - 1)]
        for i in range(self.n):
            combs = tuple(it.combinations(a, i))
            for comb in combs:
                yield list(comb) + base_list

    def all_sets_of_parametres(self):
        d_iter = list(self._generate_d_set_iterator())
        s_iter = list(self._generate_s_set_iterator())
        for d in d_iter:
            for s in s_iter:
                # for step in range(1, r):
                for step in range(1, self.r):
                    yield (d, s, step)


class InvolutiveRegisterFunction(AdditiveRegisterFunction):

    def __init__(self, D, n, r):
        super().__init__(D, n, r)
        self.middle_bit_mask = (((1 << r) - 1) ^ 1) ^ (1 << (r - 1))

    def act(self, state: list):
        last_block = 0
        for d in self.D:
            last_block = last_block + state[d] # сумма
        last_block %= (1 << self.r)
        last_block = self.swap_most_and_least(last_block)    # свапа
        next_state: list = state[1:]  # сдвиг блоков регистра
        next_state.append(last_block)
        return next_state

    def swap_most_and_least(self, a):
        most_significant_bit = a >> (self.r - 1)
        least_significant_bit = a % 2
        a &= self.middle_bit_mask
        a |= most_significant_bit
        a |= least_significant_bit << (self.r - 1)
        return a

    def analytical_create_mixing_matrix(self):
        nr = self.n * self.r
        triangle_matrix = self._create_triangle_matrix()
        triangle_matrix[:, 0] += triangle_matrix[:, self.r - 1]
        triangle_matrix[:, self.r - 1] = triangle_matrix[:, 0] - triangle_matrix[:, self.r - 1]
        triangle_matrix[:, 0] -= triangle_matrix[:, self.r - 1]

        identity_matrix = np.identity(self.r*(self.n-1), dtype=np.uint64)
        mixing_matrix = np.zeros((nr, nr), dtype=np.uint64)
        mixing_matrix[self.r: self.n * self.r, 0: (self.n - 1) * self.r] = identity_matrix
        for d in self.D:
            mixing_matrix[d * self.r: (d + 1) * self.r, (self.n - 1) * self.r: self.n * self.r] = triangle_matrix
        return mixing_matrix


class TriangleRegisterFunction(AdditiveRegisterFunction):

    def __init__(self, D, n, r):
        super().__init__(D, n, r)
        self.essential_vars = list(self._generate_essential_vars())

    def act(self, state: list):
        last_block = 0
        for d in self.D:
            last_block = last_block + state[d]     # сумма
        last_block %= (1 << self.r)
        modified_last_block = 0
        for ess_vars in self.essential_vars:
            out_bit = 0
            for ess_var in ess_vars:
                out_bit ^= ((last_block & (1 << (self.r - 1 - ess_var))) >> (self.r - 1 - ess_var))
            modified_last_block ^= (out_bit << (self.r - 1 - ess_vars[0]))
        next_state: list = state[1:]  # сдвиг блоков регистра
        next_state.append(modified_last_block)
        return next_state

    def _generate_essential_vars(self):
        for i in range(self.r):
            if i == 0:
                yield (0, )
            else:
                yield (i, 0)


class SboxV4RegisterFunction(AdditiveRegisterFunction):     # r = 4, т.к. S-блок
    #  id - tc26 - gost - 28147 - param - Z
    def __init__(self, D, n):
        super().__init__(D, n, 4)
        self.sbox = (0xC, 0x4, 0x6, 0x2, 0xA, 0x5, 0xB, 0x9, 0xE, 0x8, 0xD, 0x7, 0x0, 0x3, 0xF, 0x1)

    def act(self, state: list):
        last_block = 0
        for d in self.D:
            last_block = last_block + state[d]     # сумма
        last_block %= (1 << self.r)
        last_block = self.sbox[last_block]
        next_state: list = state[1:]  # сдвиг блоков регистра
        next_state.append(last_block)
        return next_state

class SboxV32RegisterFunction(AdditiveRegisterFunction):     # r = 32
    #  id - tc26 - gost - 28147 - param - Z
    def __init__(self, D, n):
        super().__init__(D, n, 32)
        self.sbox = (0xC, 0x4, 0x6, 0x2, 0xA, 0x5, 0xB, 0x9, 0xE, 0x8, 0xD, 0x7, 0x0, 0x3, 0xF, 0x1)

    def act(self, state: list):
        last_block = 0
        for d in self.D:
            last_block = last_block + state[d]     # сумма
        last_block %= (1 << self.r)

        last_modified_subblock = self.sbox[last_block >> 28]
        last_block <<= 4
        for i in range(32//4):
            last_block ^= (last_modified_subblock << 4*i)
        next_state: list = state[1:]  # сдвиг блоков регистра
        next_state.append(last_block)
        return next_state
