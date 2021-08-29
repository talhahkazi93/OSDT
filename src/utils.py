import gmpy2
from gmpy2 import mpz
from enum import Enum,auto
import abc

"""
    Implementation of interface for the objective function class
"""
class ObjFuncInterface(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'calc_lookahead') and
                callable(subclass.calc_lookahead) and
                hasattr(subclass, 'calc_risk') and
                callable(subclass.calc_risk) and
                hasattr(subclass, 'calc_loss') and
                callable(subclass.calc_loss) and
                hasattr(subclass, 'calc_leaf_supp') and
                callable(subclass.calc_leaf_supp) and
                hasattr(subclass, 'calc_incrm_acc') and
                callable(subclass.calc_incrm_acc) and
                hasattr(subclass, 'calc_acc_supp') and
                callable(subclass.calc_acc_supp) or
                NotImplemented)

    @abc.abstractmethod
    def calc_lookahead(self,R_c,lb,b0,n_removed_leaves,lambbb):
        """Load in the data set"""
        raise NotImplementedError

    @abc.abstractmethod
    def calc_risk(self, leaves, ext):
        """Extract text from the data set"""
        raise NotImplementedError

    @abc.abstractmethod
    def calc_loss(self, cache_tree, splitleaf,l,ext):
        """Extract text from the data set"""
        raise NotImplementedError

    @abc.abstractmethod
    def calc_leaf_supp(self,loss,len):
        """Extract text from the data set"""
        raise NotImplementedError

    @abc.abstractmethod
    def calc_incrm_acc(self,new_leaves,i):
        """Extract text from the data set"""
        raise NotImplementedError

    @abc.abstractmethod
    def calc_acc_supp(self,new_leaf,ndata):
        """Extract text from the data set"""
        raise NotImplementedError

"""
    OSDT class objective function with all related bounds
"""
class OSDT(ObjFuncInterface):

    def __init__(self,lamb):
        self.lamb = lamb

    def calc_lookahead(self,R_c,lb,b0,n_removed_leaves,lambbb):
        lookahead = False
        if lb + b0 + n_removed_leaves * lambbb >= R_c:
            lookahead = True
        return lookahead

    def calc_risk(self, leaves, ext):
        risk = sum([l.loss for l in leaves]) + self.lamb * len(leaves)
        return risk

    def calc_loss(self, cache_tree, splitleaf,l,ext):
        lb = sum([cache_tree.leaves[i].loss for i in range(l)
                       if splitleaf[i] == 0]) + self.lamb * l
        return lb

    def calc_leaf_supp(self,loss,len):
        # Osdt leaf support
        is_dead = loss <= self.lamb
        return is_dead

    def calc_incrm_acc(self,new_leaves,i):
        calcobj = self.lamb
        return calcobj

    def calc_acc_supp(self,new_leaf,ndata):
        validsupp = False
        if (new_leaf.num_captured - new_leaf.num_captured_incorrect) / ndata <= self.lamb:
            validsupp = True
        return validsupp

"""
    External path length class objective function with all related bounds
"""
class ExternalPathLength(ObjFuncInterface):

    def __init__(self,lamb):
        self.lamb = lamb

    def calc_lookahead(self,R_c,lb,b0,n_removed_leaves,lambbb):
        lookahead = False
        # External Path length
        if lb + b0 + n_removed_leaves * 2 * lambbb >= R_c:
            lookahead = True
        return lookahead

    def calc_risk(self, leaves, ext):
        #External path length
        risk = sum([l.loss for l in leaves]) + self.lamb * ext
        return risk

    def calc_loss(self, cache_tree, splitleaf,l,ext):
        # #External path length lb
        lb = sum([cache_tree.leaves[i].loss for i in range(l)
                       if splitleaf[i] == 0]) + self.lamb * ext
        return lb

    def calc_leaf_supp(self,loss,len):
        # External path length support bound
        is_dead = loss <= 2 * self.lamb * (len + 2)
        return is_dead

    def calc_incrm_acc(self,new_leaves,i):
        # External path length incremental support
        calcobj = self.lamb * (new_leaves[i].len + 2)
        return calcobj

    def calc_acc_supp(self,new_leaf,ndata):
        validsupp = False
        # External path length
        if (new_leaf.num_captured - new_leaf.num_captured_incorrect) / ndata <= self.lamb * (new_leaf.len + 1):
            validsupp = True
        return validsupp

"""
    Internal path length class objective function with all related bounds
"""
class Internalpathlength(ObjFuncInterface):

    def __init__(self,lamb):
        self.lamb = lamb

    def calc_lookahead(self,R_c,lb,b0,n_removed_leaves,lambbb):
        lookahead = False
        if lb + b0 + n_removed_leaves * lambbb >= R_c:
            lookahead = True
        return lookahead

    def calc_risk(self, leaves, ext):
        # Internal path length
        intpl = ext - 2 * (len(leaves) - 1)
        risk = sum([l.loss for l in leaves]) + self.lamb * intpl
        return risk

    def calc_loss(self, cache_tree, splitleaf,l,ext):
        # Internal path length
        intpl = ext - 2 * (l - 1)
        lb = sum([cache_tree.leaves[i].loss for i in range(l)
                       if splitleaf[i] == 0]) + self.lamb * intpl
        return lb

    def calc_leaf_supp(self,loss,len):
        # Internal path length
        is_dead = loss <= 2 * self.lamb * (len)
        return is_dead

    def calc_incrm_acc(self,new_leaves,i):
        # Internal path length
        calcobj = self.lamb * (new_leaves[i].len)
        return calcobj

    def calc_acc_supp(self,new_leaf,ndata):
        validsupp = False
        # Internal path length
        if (new_leaf.num_captured - new_leaf.num_captured_incorrect) / ndata <= self.lamb * (new_leaf.len):
            validsupp = True
        return validsupp


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
