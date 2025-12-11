The notebooks are out of date. In testing the the pybfs code, we found that it produced different results than
the R version for both the BFS function and the Build Tables function. Correct versions are in the pybfs subdirectory.
If we want to enable usage in colab, we should probably publish the package on pypi and then install/import
the package in the notebook. 