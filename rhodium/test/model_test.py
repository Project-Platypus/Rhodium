import unittest
from rhodium.model import IntegerUncertainty

class TestIntegerUncertainty(unittest.TestCase):

    def testLevels(self):
        iu = IntegerUncertainty(0, 10)
        print iu.levels(50)
