# Linear Regression Notes

27th October 2025

- The initialization of the weight matrix is deferred to the `fit` method because the dimensions of the input data are unknown at the time of initialization.
- The `predict` method is built before the `fit` method because it will be used within `fit`
- the `epochs` parameter is set at initialization instead of while fitting because it allows a separation of the hyperparameters (config) from the actual training. For deep learning, it makes more sense to set `epochs` in `train`.
- Right now I'm just focused on implementing linear regression with SGD. I will add other optimizers later.
- MSE isn't applicable right now because we are using SGD where we only consider one sample per step
- `np.expand_dims` is the numpy equivalent of `torch.unsqueeze`
- returning `self` at the end of the `fit` and `fit_normal` methods makes it similar to sklearn