from mag_utils import *
import logging

if __name__ == '__main__':

   # log_all_circuits(4, 4)
    #log_all_7_iops("logs\exps.log", 8, 32)
    # current_time = time()
    # log_name = "logs\main_" + str(current_time) + ".log"
    # logging.basicConfig(filename=log_name, level=logging.INFO, format='%(message)s')
    # logging.info("START IN {}".format(current_time))
    # ([0, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 6, 7], 31)
    # n = 8
    # r = 32
    # D = [0, 1, 2, 3, 4, 5, 6, 7]
    # S = [7]
    # shift = 2
    # # logging.info("n = {0}, r = {1}, D = {2}, S = {3}\n".format(n, r, D, S))
    # f3 = VerticalShiftRegisterFunction(D, S, shift, n, r)
    # x = find_approx_independend_vars_7(f3, 11, 12000000)
    # # logging.info("0-iop = {}".format(x))
    # for p, i in x:
    #     print("power = {0}, count of indeps = {1}".format(p, i))


    n = 8
    r = 32
    D = [0, 1, 2, 3, 4, 5, 6, 7]
    S = [7]
    shift = 2
    #
    f = InvolutiveRegisterFunction(D, n, r)
    f1 = TriangleRegisterFunction(D, n, r)
    f2 = SboxV32RegisterFunction(D, n)
    f3 = VerticalShiftRegisterFunction(D, S, shift, n, r)
    #
    # f.all_sets_of_parametres()
    #log_all_exps(f, n, r)
    register_function_runtime_logger((f, f1, f2, f3), n, r, 50000, 5)

