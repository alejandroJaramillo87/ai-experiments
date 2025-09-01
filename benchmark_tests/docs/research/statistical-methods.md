# Statistical Methods

Advanced statistical techniques for language model evaluation and comparison.

## Hypothesis Testing Framework

### Model Comparison Testing

**Definition 1.1** (Evaluation Hypothesis Framework):
For models M₁, M₂ and evaluation metric μ:
- H₀: μ(M₁) = μ(M₂) (no performance difference)
- H₁: μ(M₁) ≠ μ(M₂) (significant performance difference)

**Test Statistic**:
For evaluation scores X₁ = {x₁₁, x₁₂, ..., x₁ₙ} and X₂ = {x₂₁, x₂₂, ..., x₂ₘ}:

t = (X̄₁ - X̄₂) / √(s²ₚ(1/n + 1/m))

where s²ₚ = ((n-1)s₁² + (m-1)s₂²) / (n+m-2) is the pooled variance.

**Statistical Power Analysis**:
Required sample size for detecting effect size d with power 1-β:
n ≥ 2(z_{α/2} + z_β)² / d²

**Implementation** (`statistical_accuracy_validation.py:45-78`):
- Welch's t-test for unequal variances
- Bootstrap confidence intervals for non-normal distributions
- Effect size calculation using Cohen's d

### Multiple Comparison Corrections

**Bonferroni Correction**:
For k simultaneous comparisons, adjusted significance level:
α_adjusted = α / k

**Benjamini-Hochberg Procedure**:
For controlling False Discovery Rate at level q:
1. Order p-values: p₍₁₎ ≤ p₍₂₎ ≤ ... ≤ p₍ₖ₎
2. Find largest i such that p₍ᵢ₎ ≤ (i/k)q
3. Reject hypotheses 1, 2, ..., i

**Holm-Bonferroni Method**:
Sequentially test ordered p-values against α/(k+1-i).

## Bayesian Evaluation Framework

### Bayesian Model Comparison

**Definition 2.1** (Bayesian Model Evaluation):
For models M = {M₁, M₂, ..., Mₖ} and evaluation data D:

P(Mᵢ|D) = P(D|Mᵢ)P(Mᵢ) / ∑ⱼ P(D|Mⱼ)P(Mⱼ)

**Bayes Factor**:
BF₁₂ = P(D|M₁) / P(D|M₂) = [P(M₁|D)/P(M₂|D)] × [P(M₂)/P(M₁)]

**Evidence Interpretation**:
- BF > 10: Strong evidence for M₁
- 3 < BF < 10: Moderate evidence for M₁  
- 1/3 < BF < 3: Weak evidence
- BF < 1/3: Evidence against M₁

### Hierarchical Bayesian Modeling

**Model Structure**:
For evaluation scores yᵢⱼ (model i, test j):

yᵢⱼ ~ Normal(μᵢⱼ, σ²)
μᵢⱼ = αᵢ + βⱼ + γᵢⱼ
αᵢ ~ Normal(μ_α, σ²_α)  [model effects]
βⱼ ~ Normal(μ_β, σ²_β)  [test effects]
γᵢⱼ ~ Normal(0, σ²_γ)   [interaction effects]

**Prior Specifications**:
- μ_α ~ Normal(0.5, 0.2²)  [weakly informative]
- σ²_α ~ InvGamma(2, 1)
- Similar for β and γ parameters

**MCMC Implementation**:
- Hamiltonian Monte Carlo with NUTS sampler
- 4 chains, 2000 iterations each, 1000 warmup
- Convergence diagnostics: R̂ < 1.1, effective sample size > 400

## Non-Parametric Methods

### Rank-Based Testing

**Wilcoxon Signed-Rank Test**:
For paired comparisons of models on same test set:

W = ∑ᵢ I(dᵢ > 0) × rank(|dᵢ|)

where dᵢ = score₁ᵢ - score₂ᵢ

**Mann-Whitney U Test**:
For independent samples from different models:

U₁ = n₁n₂ + n₁(n₁+1)/2 - R₁

where R₁ is sum of ranks for first sample.

**Kruskal-Wallis Test**:
For comparing k > 2 models:

H = (12/N(N+1)) × ∑ᵢ (R²ᵢ/nᵢ) - 3(N+1)

where Rᵢ is rank sum for group i.

### Bootstrap Methods

**Algorithm 1** (Bootstrap Confidence Intervals):
```
Input: evaluation scores X = {x₁, x₂, ..., xₙ}, confidence level 1-α
Output: confidence interval [L, U]

function bootstrap_ci(X, α, B=10000):
    bootstrap_means = []
    for b in 1 to B:
        X_bootstrap = sample_with_replacement(X, n)
        bootstrap_means.append(mean(X_bootstrap))
    
    L = quantile(bootstrap_means, α/2)
    U = quantile(bootstrap_means, 1-α/2)
    return [L, U]
```

**Bias-Corrected and Accelerated (BCa) Bootstrap**:
Adjusts for bias and skewness in bootstrap distribution:

α₁ = Φ(ẑ₀ + (ẑ₀ + z_{α/2})/(1 - â(ẑ₀ + z_{α/2})))
α₂ = Φ(ẑ₀ + (ẑ₀ + z_{1-α/2})/(1 - â(ẑ₀ + z_{1-α/2})))

where ẑ₀ is bias-correction and â is acceleration parameter.

## Time Series Analysis for Longitudinal Evaluation

### Autoregressive Models

**AR(p) Model for Evaluation Scores**:
Xₜ = c + φ₁Xₜ₋₁ + φ₂Xₜ₋₂ + ... + φₚXₜ₋ₚ + εₜ

where εₜ ~ Normal(0, σ²).

**Model Selection**:
Use AIC/BIC for optimal lag order:
AIC = 2k - 2ln(L)
BIC = k×ln(n) - 2ln(L)

**Stationarity Testing**:
Augmented Dickey-Fuller test:
ΔXₜ = α + βt + γXₜ₋₁ + ∑δᵢΔXₜ₋ᵢ + εₜ

### Changepoint Detection

**CUSUM Method**:
For detecting changes in evaluation score means:

Sₜ = ∑ᵢ₌₁ᵗ (Xᵢ - X̄)

**PELT Algorithm** (Pruned Exact Linear Time):
Optimal changepoint detection with complexity O(n log n).

## Multi-Level Modeling

### Cultural Context as Random Effect

**Model Specification**:
For evaluation scores with cultural contexts:

yᵢⱼₖ = β₀ + β₁×modelᵢ + β₂×testⱼ + u₀ₖ + εᵢⱼₖ

where:
- u₀ₖ ~ Normal(0, σ²ᵤ) is random intercept for culture k
- εᵢⱼₖ ~ Normal(0, σ²) is residual error

**Intraclass Correlation**:
ICC = σ²ᵤ / (σ²ᵤ + σ²)

**Likelihood Ratio Test**:
Compare models with/without random effects using:
LRT = -2(log L₀ - log L₁) ~ χ²₁

### Cross-Classified Models

**Model for Crossed Random Effects**:
When tests and cultures are crossed:

yᵢⱼₖ = β₀ + β₁×modelᵢ + u₀ⱼ + v₀ₖ + εᵢⱼₖ

where u₀ⱼ and v₀ₖ are independent random effects.

## Measurement Theory

### Classical Test Theory

**True Score Model**:
X = T + E

where:
- X is observed score
- T is true score  
- E is measurement error

**Reliability**:
ρ = Var(T) / Var(X) = (σ²ₜ) / (σ²ₜ + σ²ₑ)

**Cronbach's Alpha**:
α = (k/(k-1)) × (1 - ∑σ²ᵢ/σ²ₜₒₜₐₗ)

### Item Response Theory

**2-Parameter Logistic Model**:
P(Xᵢⱼ = 1|θⱼ, aᵢ, bᵢ) = 1 / (1 + exp(-aᵢ(θⱼ - bᵢ)))

where:
- θⱼ is model ability
- aᵢ is item discrimination  
- bᵢ is item difficulty

**Parameter Estimation**:
Use Marginal Maximum Likelihood with EM algorithm:

E-step: Q(Ψ|Ψ⁽ᵗ⁾) = E[log L(Ψ)|X, Ψ⁽ᵗ⁾]
M-step: Ψ⁽ᵗ⁺¹⁾ = argmax Q(Ψ|Ψ⁽ᵗ⁾)

## Experimental Design

### Factorial Designs

**2^k Factorial Design**:
For k evaluation factors, requires 2^k conditions.

**Fractional Factorial Design**:
2^{k-p} design with confounding structure defined by generators.

**Response Surface Methodology**:
Second-order model:
y = β₀ + ∑βᵢxᵢ + ∑βᵢᵢx²ᵢ + ∑∑βᵢⱼxᵢxⱼ + ε

### Latin Square Designs

**Structure**:
Block by two nuisance factors (test order, evaluator):

Model: yᵢⱼₖ = μ + αᵢ + βⱼ + γₖ + τₗ₍ᵢⱼ₎ + εᵢⱼₖ

where τₗ₍ᵢⱼ₎ is treatment effect for treatment in position (i,j).

## Meta-Analysis Methods

### Fixed Effects Model

**Weighted Average**:
θ̂ = (∑wᵢθ̂ᵢ) / (∑wᵢ)

where wᵢ = 1/σ²ᵢ

**Test of Homogeneity**:
Q = ∑wᵢ(θ̂ᵢ - θ̂)² ~ χ²ₖ₋₁

### Random Effects Model

**DerSimonian-Laird Estimator**:
τ̂² = max(0, (Q-(k-1))/(∑wᵢ - ∑w²ᵢ/∑wᵢ))

**Prediction Interval**:
θ̂ ± t_{k-2,α/2} × √(v̂ + τ̂²)

## Computational Methods

### Monte Carlo Methods

**Importance Sampling**:
For intractable posterior p(θ|y):

E[g(θ)|y] ≈ (∑g(θᵢ)w(θᵢ)) / (∑w(θᵢ))

where w(θᵢ) = p(θᵢ|y)/q(θᵢ) and θᵢ ~ q(θ).

**Acceptance-Rejection Sampling**:
For sampling from complex distributions using envelope function.

### Variational Inference

**Mean Field Approximation**:
Approximate p(θ|y) with q(θ) = ∏q_j(θⱼ).

**Evidence Lower Bound (ELBO)**:
L(q) = E_q[log p(y,θ)] - E_q[log q(θ)]

Optimize: q* = argmax L(q)

## References

- [Theoretical Foundations](./theoretical-foundations.md)
- [Evaluation Algorithms](./evaluation-algorithms.md)
- [Experimental Validation](./experimental-validation.md)