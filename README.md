# SODA 

Code accompanying the paper [Optimistic Dual Averaging Unifies Modern Optimizers](https://arxiv.org/pdf/2605.11172).


## Repository Structure

- [`soda.py`](soda.py): Contains the `SODA` reference implementation along with various norm choices.
- [`moda.py`](moda.py): Contains `MODA`, the Modernized Optimistic Dual Averaging special case for which the implementation simplifies.
- [`soda_wrapper.py`](soda_wrapper.py): A lightweight wrapper that adds the SODA averaging step on top of an existing optimizer without weight decay.
- [`modded-nanogpt/`](modded-nanogpt): Example usage of nanoGPT experiments.


## Documentation

### SODA

$$
\begin{aligned}
m^{k+1} &= (1-\alpha_k)m^k+\alpha_k\nabla f(y^k,\xi_k),\\
\bar m^{k+1} &= (1-\bar\alpha_k)m^{k+1}+\bar\alpha_k\nabla f(y^k,\xi_k)
\quad \text{(optimism)}\\
z^{k+1} &\in \partial h_k^*(-\eta_k(k+2)\bar m^{k+1}) \\
x^{k+1} &= (1-\lambda_k)x^k+\lambda_k z^{k+1},\\
y^{k+1} &= (1-\bar\lambda_k)x^{k+1}+\bar\lambda_k z^{k+1}
\qquad \quad\quad\  \text{(primal extrapolation)}
\end{aligned}
$$

The `SODA` optimizer comes with the following hyperparameters:

| Hyperparameter     | Meaning                                                | Default in `soda.py` | Common setting / comment                                                                                                 |
| ------------------ | ------------------------------------------------------ | -------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `lr`               | Learning-rate $\eta_k$.                                | `1e-4`               | Typically uses a schedule.                                                                                               |
| `dual_momentum1`   | Dual averaging parameter $\alpha_k$.                   | `0.05`               | This is `1 - usual_momentum` in PyTorch SGD-style notation. Same default as in Muon.                                     |
| `dual_momentum2`   | Optimistic parameter $\bar\alpha_k$.                   | `0.05`               | Uses the same `1 - usual_momentum` convention and default. Set to `0` to disable optimism.                               |
| `primal_momentum1` | Averaging parameter $\lambda_k$ for $x^{k+1}$.         | `1 / (k + 2)`        | Uniform averaging over the primal iterates.                                                                              |
| `primal_momentum2` | Extrapolation parameter $\bar\lambda_k$ for $y^{k+1}$. | `0`                  | Default enables primal extrapolation, so gradients and evaluation use the same iterate. This is the `MODA` special case. |
| `norm`             | Norm choice for $h_k$.                                 | `"Spectral"`         | Supported: `Spectral`, `SpectralConv`, `Sign`, `ColNorm`, `RowNorm`, `BiasRMS`, `Sinkhorn`.                              |
| `norm_kwargs`      | Arguments for the selected norm.                       | `{}`                 | For example, `{"steps": 5}` for `Spectral` or `SpectralConv`.                                                            |

> **Note** For evaluation with standalone `SODA`, call `optimizer.eval()` before validation and `optimizer.train()` before resuming training. This switches between the averaged iterate $x^k$, where performance is evaluated, and the extrapolated iterate $y^k$, where gradients are computed. With the default $\bar\lambda_k=0$, this simplifies to $x^k=y^k$ (see `MODA`).

**Usage**:

```python
optim_groups = [{
    "params": model.transformer.h.parameters(),
    "lr": 50 * 2**-12,
    "norm": "Spectral",
    "norm_kwargs": {"steps": 5},
}, {
    "params": model.lm_head.parameters(),
    "lr": 3000 * 2**-12,
    "norm": "Sign",
    "norm_kwargs": {},
}]

optimizer = SODA(
    optim_groups,
    dual_momentum1=0.05,
    dual_momentum2=0.05,
)
```

These parameter choices are based on the hyperparameters of uScion, which surprisingly works out of the box even with the 1/k schedule in `SODA`.

### MODA

`MODA` is the Modernized Optimistic Dual Averaging special case of `SODA`, obtained by setting the primal extrapolation parameter to zero (`primal_momentum2=0`).
This configuration corresponds to evaluating the performance and the gradients on the same model, which simplifies the code substantially since `optimizer.train()` and `optimizer.eval()` can be avoided.

**Usage**:

```python
optimizer = MODA(
    optim_groups,
    dual_momentum1=0.05,
    dual_momentum2=0.05,
)
```

### SODAWrapper

$$
x^{k+1}=
\underbrace{\frac{1}{k+2}z^0}_{\text{center at init.}}
+\underbrace{\left(1-\frac{1}{k+2}\right)x^k}_{\text{\(1/k\) decay}}
+\underbrace{\text{BaseUpdate}(g^k)}_{\text{base step}}.
$$

`SODAWrapper` has no additional hyperparameters. It wraps a base optimizer and applies the SODA averaging step after the base optimizer update. The wrapper should be used with weight decay disabled in the base optimizer.

An existing optimizer can be wrapped as:

```python
base_optimizer = Muon(params, lr=lr, weight_decay=0)
optimizer = SODAWrapper(base_optimizer)
```

> **Note** The purpose of the SODA Wrapper is to quickly test on any given base optimizer. It is not memory optimized and for production we recommend folding the wrapper logic into the base optimizer.


## Citation

If you find this work useful, please cite it as follows:

```bibtex
@article{pethick2026optimistic,
  title={Optimistic Dual Averaging Unifies Modern Optimizers},
  author={Pethick, Thomas and Xie, Wanyun and Machacek, Roman and Cevher, Volkan},
  journal={arXiv preprint arXiv:2605.11172},
  year={2026}
}
```
