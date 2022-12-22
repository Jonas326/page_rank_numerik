import unittest
import numpy as np
import PageRank


def e(index, dim=12):
    # method to create standard base vectors
    arr = np.zeros(dim)
    arr[index - 1] = 1
    return arr


class MyTestCase(unittest.TestCase):
    import numpy as np
    import PageRank

    L_tilde = np.array([e(2) + e(3), e(5), e(1) + e(2), e(5), e(4) + e(7) + e(9),
                        e(5), np.ones(12), e(5), e(6), e(8), e(8), e(8)])
    L_diagonal_inverse = np.array([e(1) * 1 / 2, e(2), e(3) * 1 / 2, e(4), e(5) * 1 / 3,
                                   e(6), e(7) * 1 / 12, e(8), e(9), e(10), e(11), e(12)])

    def test_page_rank(self):
        self.np.testing.assert_array_almost_equal(PageRank.tilde(PageRank.L), self.L_tilde)
        self.np.testing.assert_array_almost_equal(PageRank.diagonal_inverse(self.L_tilde), self.L_diagonal_inverse)


if __name__ == '__main__':
    unittest.main()
