Welcome to shapicant's documentation!
=====================================

**shapicant** is a feature selection package based on `SHAP <https://github.com/slundberg/shap>`_ [LUN]_ and target permutation, for pandas and Spark.

It is inspired by PIMP [ALT]_, with some differences:

- PIMP fits a probability distribution to the population of null importances or, alternatively, uses a non-parametric estimation of the PIMP p-values. Instead, shapicant only implements the non-parametric estimation.
- For the non-parametric estimation, PIMP computes the fraction of null importances that are more extreme than the true importance (i.e. :code:`r/n`). Instead, shapicant computes it as :code:`(r+1)/(n+1)` [NOR]_.
- PIMP uses the Gini importance of Random Forest models or the Mutual Information criterion. Instead, shapicant uses SHAP values.
- While feature importance measures such as the Gini importance show an absolute feature importance, SHAP provides both positive and negative impacts. Instead of taking the mean absolute value of the SHAP values for each feature as feature importance, shapicant takes the mean value for positive and negative SHAP values separately. The true importance needs to be consistently higher than null importances for both positive and negative impacts. For multi-class classification, the true importance needs to be higher for at least one of the classes.
- While feature importance measures such as the Gini importance of Random Forest models are computed on the training set, SHAP values can be computed out-of-sample. Therefore, shapicant allows to compute them on a distinct validation set. To decide whether to compute them on the training set or on a validation set, you can refer to this discussion for "`Training vs. Test Data <https://compstat-lmu.github.io/iml_methods_limitations/pfi-data.html>`_" (it talks about PFI [BRE]_, which is a different algorithm, but the general idea is still applicable).

Permuting the response vector instead of permuting features has some advantages:

- The dependence between predictor variables remains unchanged.
- The number of permutations can be much smaller than the number of predictor variables for high dimensional datasets (unlike PFI [BRE]_) and there is no need to add shadow features (unlike Boruta [KUR]_).
- Since the features set does not change during iterations, the distributed implementation is more straightforward.

References
==========

.. [LUN] Lundberg, S., & Lee, S.I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems* (pp. 4765–4774).
.. [ALT] Altmann, A., Toloşi, L., Sander, O., & Lengauer, T. (2010). Permutation importance: a corrected feature importance measure *Bioinformatics, 26* (10), 1340-1347.
.. [NOR] North, B. V., Curtis, D., & Sham, P. C. (2002). A note on the calculation of empirical P values from Monte Carlo procedures. *American journal of human genetics, 71* (2), 439–441.
.. [BRE] Breiman, L. (2001). Random Forests *Machine Learning, 45* (1), 5–32.
.. [KUR] Kursa, M., & Rudnicki, W. (2010). Feature Selection with Boruta Package *Journal of Statistical Software, 36*, 1-13.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   examples
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
