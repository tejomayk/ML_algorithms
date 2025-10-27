# Linear Regression

27th October 2025

- The initialization of the weight matrix is deferred to the `fit` method because the dimensions of the input data are unknown at the time of initialization.
- The `predict` method is built before the `fit` method because it will be used within `fit`