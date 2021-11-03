# Optimal Sparse Decision Trees Revisited (OSDTR)
This work uses an alternate approach to the work of ["Optimal Sparse Decision Trees"](https://arxiv.org/abs/1904.12847) by Xiyang Hu,
Cynthia Rudin, and Margo Seltzer. We uses differnt objective function to find the optimal decision tree.

### Dependencies

* [gmp](https://gmplib.org/) (GNU Multiple Precision Arithmetic Library)
* [mpfr](http://www.mpfr.org/) (GNU MPFR Library for multiple-precision floating-point computations; depends on gmp)
* [libmpc](http://www.multiprecision.org/) (GNU MPC for arbitrarily high precision and correct rounding; depends on gmp and mpfr)
* [gmpy2](https://pypi.org/project/gmpy2/#files) (GMP/MPIR, MPFR, and MPC interface to Python 2.6+ and 3.x)

### Main function
The main function is the `bbound()` function in `osdtr.py`.

### Arguments
**[objective_function]** Select objective function with sparsity panelty from.
  * Use `Encodings.AsymEnc` for asymmetric binary encoding scheme.
  * Use `Encodings.GenericEnc`  for generic binary encoding scheme.
  * Use `Encodings.NumBasedEnc`  for number-based binary encoding scheme.
  * Use `Encodings.OsdtEnc`  for one-hot(modified-asymmetric) binary encoding scheme.

**[encoding]** Select encoding scheme to convert dataset into binary from.
  * Use `ObjFunction.OSDT` for number of leaves as sparsity panelty.
  * Use `ObjFunction.ExternalPathLength` for External path length as sparsity panelty.
  * Use `ObjFunction.InternalPathLength` for Internal path length as sparsity panelty.
  * Use `ObjFunction.WeightedExternalPathLength` for Weighted external path length as sparsity panelty.


**[x]** The features of the training data.

**[y]** The labels of the training data.

**[lamb]** The regularization parameter `lambda` of the objective function.

**[prior_metric]** The scheduling policy.

* Use `curiosity` to prioritize by curiosity.

**[MAXDEPTH]** Maximum depth of the tree. Default value is `float('Inf')`.

**[MAX_NLEAVES]** Maximum number of leaves of the tree. Default value is `float('Inf')`.

**[niter]** Maximum number of tree evaluations. Default value is `float('Inf')`.

**[support]** Turn on `Lower bound on leaf support`. Default is `True`.

**[incre_support]** Turn on `Lower bound on incremental classification accuracy`. Default is `True`.

**[accu_support]** Turn on `Lower bound on classification accuracy`. Default is `True`.

**[equiv_points]** Turn on `Equivalent points bound`. Default is `True`.

**[lookahead]** Turn on `Lookahead bound`. Default is `True`.

**[timelimit]** Time limit on the running time. Default is `True`.


### Example test code

We provide our test code in `run_test.py`.

### Dataset

See `data/datasets/`.

We used 7 datasets: Five of them are from the UCI Machine Learning Repository (tic-tac-toc, car evaluation, monk1, monk2, monk3). 
The other two datasets are the ProPublica recidivism data set and the Fair Isaac (FICO) credit risk datasets. 
We predict which individuals are arrested within two years of release (`{N = 7,215}`) on the recidivism data set and whether an individual will default on a loan for the FICO dataset. There are also diffrent encoding styles for the five UCI datasets corresponding to their respective folders. 
