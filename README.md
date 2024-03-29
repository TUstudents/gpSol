# A Bayesian Approach to Predict Solubility Parameters

The critical role of solubility spans numerous domains, affecting everything from liquid miscibility and polymer stability to solid adsorption. This phenomenon’s accurate and swift prediction can significantly advance a variety of industries, including organic semiconductors, paint coatings, pharmaceuticals, and every chemical synmthesis. However, the challenge in predicting solubility lies in the complex interplay between the solute and solvent, compounded by a myriad of other physical and chemical properties. This complexity underscores the need for advanced methodologies that can navigate the intricacies of solubility, thereby enabling better material design and formulation across multiple fields.

In [1] Gaussian Process Modeling was proposed to predict the Hansen solubility parameter from chemical descriptors and sigma profiles. However, not the complete code of the implementation of the paper was provided, but it includes a huge dataset and can serve as benchmark.
Recently, GAUCHE [2] was released, a framework for gaussian processes in chemistry. It builds on top of gpytorch, which gives it the GPU computation ability of Torch and Bayesian optimization implemented in botorch. It includes the interfaces to over 30 kernels, including kernels specific for molecular fingerprints.

Here, the dataset of [1] is modeled with GP and the performance of different kernels compared.


## Why using GAUCHE?

General-purpose Gaussian process (GP) and Bayesian optimisation (BO) libraries do not cater for molecular representations. Likewise, general-purpose molecular machine learning libraries do not consider GPs and BO. To bridge this gap, **GAUCHE** provides a modular, robust and easy-to-use framework of 30+ parallelisable and batch-GP-compatible implementations of string, fingerprint and graph kernels that operate on a range of widely-used molecular representations. **GAUCHE** is a collaborative, open-source software library that aims to make state-of-the-art
probabilistic modelling and black-box optimisation techniques more easily accessible to scientific
experts in chemistry, materials science and beyond.

### Kernels

Standard GP packages typically assume continuous input spaces of low and fixed dimensionality. This makes it difficult to apply them to common molecular representations: molecular graphs are discrete objects, SMILES strings vary in length and topological fingerprints tend to be high-dimensional and sparse. To bridge this gap, GAUCHE provides:

* **Fingerprint Kernels** that measure the similarity between bit/count vectors of descriptor by examining the degree to which their elements overlap.
* **String Kernels** that measure the similarity between strings by examining the degree to which their sub-strings overlap.
* **Graph Kernels** that measure between graphs by examining the degree to which certain substructural motifs overlap.

---

## Baseline

Baseline according to [1] [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TUstudents/gpSol/blob/main/notebooks/Baseline_Implementation.ipynb)

---

### Example Usage: Loading and Featurising Molecules

GAUCHE provides a range of helper functions for loading and preprocessing datasets for molecular property and reaction yield prediction and optimisation tasks. For more detail, check out our [Loading and Featurising Molecules Tutorial](https://leojklarner.github.io/gauche/notebooks/loading_and_featurising_molecules.html) and the corresponding section in the [Docs](https://leojklarner.github.io/gauche/notebooks/loading_and_featurising_molecules.html).



```python	
from gauche.dataloader import MolPropLoader

loader = MolPropLoader()

# load one of the included benchmarks
loader.load_benchmark("Photoswitch")

# or a custom dataset
loader.read_csv(path="data.csv", smiles_column="smiles", label_column="y")

# and quickly featurise the provided molecules
loader.featurize('ecfp_fragprints')
X, y = loader.features, loader.labels
```

### Example Usage: GP Regression on Molecules

Fitting a GP model with a kernel from GAUCHE and using it to predict the properties of new molecules is as easy as this. For more detail, check out our [GP Regression on Molecules Tutorial](https://leojklarner.github.io/gauche/notebooks/gp_regression_on_molecules.html) and the corresponding section in the [Docs](https://leojklarner.github.io/gauche/modules/dataloader.html).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leojklarner/gauche/blob/main/notebooks/GP%20Regression%20on%20Molecules.ipynb)

```python
import gpytorch
from botorch import fit_gpytorch_model
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel

# define GP model with Tanimoto kernel
class TanimotoGP(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood):
    super(TanimotoGP, self).__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
  
  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialise GP likelihood, model and 
# marginal log likelihood objective
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = TanimotoGP(X_train, y_train, likelihood)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# fit GP with BoTorch in order to use
# the LBFGS-B optimiser (recommended)
fit_gpytorch_model(mll)

# use the trained GP to get predictions and 
# uncertainty estimates for new molecules
model.eval()
likelihood.eval()
preds = model(X_test)
pred_means, pred_vars = preds.mean, preds.variance
```




## References

[1] Sanchez-Lengeling, Roch, Perea, Langner, Brabec, Aspuru-Guzik (2018), ["A Bayesian Approach to Predict Solubility Parameters"](http://doi.org/10.1002/adts.201800069), Advanced Theory and simulations, 2018, Vol. 2, Issue 1.

[2] Griffiths, Klarner, Moss, Ravuri, Truong, Rankovic, Du, Schwartz, Jamasb, Tripp, Kell, Bourached, Chan, Guo, Lee, Schwaller and Tang, 2022. [GAUCHE: A Library for Gaussian Processes in Chemistry](https://arxiv.org/abs/2212.04450). arXiv. [GAUCHE Documentation](https://leojklarner.github.io/gauche/)


