import unittest
from main import *


class TestModifiedGeneratorMethods(unittest.TestCase):
    def setUp(self):
        self.func = VerticalShiftFunction([0, 1, 2, 3], [2,3], 1, n=4, r=4)
        self.initial_state = 59081      #1110 0110 1100 1001
        self.acted_state = 27699       #0110 1100 0011 0011
        self.acted_twice_state = 50017      #1100 0011 0110 0001
        self.acted_three_times_state = 13868      #001 0110 0010 1100
        '''''
        1110
        0110
        1100
        1001

        0110
        1100
        0011
        0011

        1100
        0011
        0110
        0001

        0011
        0110
        0010
        1100
        '''''
    def test_act(self):
      self.assertEqual(self.func.act(self.initial_state), self.acted_state)

    def test_2_times_act(self):
      self.assertEqual(self.func.act_k_times(self.initial_state, 2), self.acted_twice_state)

    def test_3_times_act(self):
      self.assertEqual(self.func.act_k_times(self.initial_state, 3), self.acted_three_times_state)
    #
    # def test_isupper(self):
    #   self.assertTrue('FOO'.isupper())
    #   self.assertFalse('Foo'.isupper())
    #
    # def test_split(self):
    #   s = 'hello world'
    #   self.assertEqual(s.split(), ['hello', 'world'])
    #   # Проверим, что s.split не работает, если разделитель - не строка
    #   with self.assertRaises(TypeError):
    #       s.split(2)

if __name__ == '__main__':
    unittest.main()