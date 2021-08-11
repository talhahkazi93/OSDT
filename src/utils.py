import gmpy2
from gmpy2 import mpz
from enum import Enum,auto

# class Encodings(Enum):
#     # Options of encoding defined
#     NumberBased = auto()
#     Asymmetric = auto()
#     ModifiedAsymmetric = auto()
#     GenericEncoding = auto()


class ObjFunction(Enum):
    # Options of possible objective function defined
    OSDT = auto()
    ExternalPathLength = auto()
    InternalPathLength = auto()

class RulesFunctions:

    """
        Python implementation of make_all_ones

        Returns a mpz object consisting of length ones

        Note: in order to ensure you have a leading one, pass in
        a length that is 1 greater than your number of samples
    """
    @classmethod
    def make_all_ones(self,length):
        ones = pow(2, length) - 1
        default_tt = mpz(ones)
        return default_tt

    """
        Python implementation of rule_vectompz
    
        Convert a binary vector to a mpz object
    
        Note: in order to ensure you have a leading one,
        add '1' in the front
    """
    @classmethod
    def rule_vectompz(self,vec):
        return mpz('1' + "".join([str(i) for i in vec]), 2)

    """
        Python implementation of rule_vand
    
        Takes in two truthtables
        Returns the and of the truthtables
        as well as the number of ones in the and
    """
    @classmethod
    def rule_vand(self,tt1, tt2):
        vand = tt1 & tt2
        # subtract 1 to remove leading ones
        cnt = gmpy2.popcount(vand) - 1
        return vand, cnt

    """
        Count the number of ones in an mpz object
        ensuring we strip off the leading one
    """
    @classmethod
    def count_ones(self,tt):
        return gmpy2.popcount(tt) - 1
