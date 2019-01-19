from mag_utils import *

if __name__ == '__main__':

    n = 4
    r = 4
    D = [0, 1, 2, 3]
    S = [3]
    shift = 1

    f = InvolutiveRegisterFunction(D, n, r)
    f1 = TriangleRegisterFunction(D, n, r)
    f2 = SboxV4RegisterFunction(D, n)
    f3 = VerticalShiftRegisterFunction(D, S, shift, n, r)

    register_function_runtime_logger((f, f1, f2, f3), n, r, 1, 1)

