import unittest

from utils.edge import *


class ToolsTestCase(unittest.TestCase):
    def test_consecutive(self):
        result = consecutive([1, 2, 3, 5, 6, 7])
        answer = [np.array([1, 2, 3]), np.array([5, 6, 7])]

        np.testing.assert_array_equal(result, answer)


if __name__ == '__main__':
    unittest.main()
