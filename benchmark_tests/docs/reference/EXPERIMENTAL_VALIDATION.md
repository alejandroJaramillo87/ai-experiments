# Experimental Validation and Statistical Analysis

This document presents the experimental methodology, statistical validation procedures, and empirical analysis frameworks used to validate the AI Model Evaluation Framework. It follows rigorous experimental design principles and advanced statistical methods appropriate for scientific research.

## 🔬 **Experimental Design Framework**

### **1. Controlled Experimental Conditions**

**Definition 1.1** (Experimental Control Variables):
Let $\mathcal{C} = \{C_1, C_2, ..., C_k\}$ be the set of controlled variables in the evaluation experiment:

- $C_1$: Model architecture (transformer type, parameter count)
- $C_2$: Training data composition (domain, size, quality)  
- $C_3$: Evaluation environment (hardware, software versions)
- $C_4$: Test case selection (difficulty, cultural context)
- $C_5$: Evaluator configuration (weights, thresholds)

**Experimental Protocol 1.1** (Standardized Evaluation Procedure):
```
1. Environment Setup:
   - Initialize identical hardware configuration
   - Set deterministic random seeds: numpy.seed(42), torch.manual_seed(42)
   - Clear system caches and normalize resource usage
   - Record baseline performance metrics

2. Model Preparation:
   - Load pre-trained model with fixed parameters
   - Disable stochastic elements (dropout, random sampling)
   - Verify model checksums for consistency
   - Warm-up inference (5 dummy evaluations)

3. Test Execution:
   - Randomize test order using fixed permutation seed
   - Execute evaluations with controlled timing
   - Monitor resource usage continuously
   - Record all intermediate states

4. Data Collection:
   - Capture raw model responses
   - Log evaluation intermediate steps
   - Record performance metrics
   - Save complete system state
```

### **2. Experimental Validity Framework**

**Definition 1.2** (Validity Threats and Mitigation):

**Internal Validity Threats:**
- **Selection bias**: Mitigated through stratified random sampling of test cases
- **Instrumentation effects**: Controlled through standardized evaluation procedures
- **Temporal variations**: Addressed via repeated measures design with time blocking

**External Validity Threats:**
- **Population validity**: Ensured through diverse model architectures and domains
- **Ecological validity**: Maintained through realistic evaluation scenarios
- **Treatment interaction**: Controlled through factorial experimental design

**Construct Validity Threats:**
- **Inadequate operationalization**: Validated through expert panel review
- **Mono-method bias**: Addressed through triangulation with multiple evaluation approaches
- **Evaluation reactivity**: Minimized through blind evaluation protocols

## 📊 **Statistical Analysis Methodology**

### **1. Power Analysis and Sample Size Determination**

**Theorem 2.1** (Required Sample Size for Effect Detection):
For detecting a minimum effect size $\delta$ with power $1-\beta$ and significance level $\alpha$ in a two-sample comparison:

$$n = \frac{2(z_{\alpha/2} + z_{\beta})^2 \sigma^2}{\delta^2}$$

**Empirical Power Calculation**:
```python
import scipy.stats as stats
import numpy as np

def calculate_required_sample_size(effect_size, power=0.8, alpha=0.05, 
                                 population_std=0.15):
    """
    Calculate required sample size for evaluation experiments.
    
    Based on pilot studies showing evaluation score std ≈ 0.15
    """
    
    # Convert power and alpha to z-scores
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate required sample size per group
    n_required = (2 * (z_alpha + z_beta)**2 * population_std**2) / effect_size**2
    
    # Add 20% buffer for dropouts/failures
    n_conservative = int(np.ceil(n_required * 1.2))
    
    return n_conservative

# Example: Detect 0.1 point difference in evaluation scores
required_n = calculate_required_sample_size(
    effect_size=0.1,    # 10% difference in normalized scores
    power=0.9,          # 90% power
    alpha=0.01          # 1% significance level
)
print(f"Required sample size per group: {required_n}")
```

### **2. Bayesian Statistical Analysis**

**Definition 2.1** (Bayesian Evaluation Model):
We model evaluation scores using a hierarchical Bayesian framework:

$$\begin{align}
y_{ijk} &\sim \mathcal{N}(\mu_{ijk}, \sigma^2) \\
\mu_{ijk} &= \alpha + \beta_i + \gamma_j + \delta_k + (\beta\gamma)_{ij} + \epsilon_{ijk} \\
\beta_i &\sim \mathcal{N}(0, \tau_\beta^2) \quad \text{(Model effect)} \\
\gamma_j &\sim \mathcal{N}(0, \tau_\gamma^2) \quad \text{(Domain effect)} \\
\delta_k &\sim \mathcal{N}(0, \tau_\delta^2) \quad \text{(Evaluator effect)} \\
\end{align}$$

**Bayesian Implementation**:
```python
import pymc3 as pm
import numpy as np
import pandas as pd

class BayesianEvaluationAnalysis:
    """
    Bayesian analysis framework for evaluation data.
    Uses MCMC sampling for robust inference under uncertainty.
    """
    
    def __init__(self):
        self.model = None
        self.trace = None
        
    def build_hierarchical_model(self, data):
        """Build hierarchical Bayesian model for evaluation analysis."""
        
        # Extract factors
        model_ids = pd.Categorical(data['model']).codes
        domain_ids = pd.Categorical(data['domain']).codes  
        evaluator_ids = pd.Categorical(data['evaluator']).codes
        
        n_models = len(data['model'].unique())
        n_domains = len(data['domain'].unique())
        n_evaluators = len(data['evaluator'].unique())
        
        with pm.Model() as hierarchical_model:
            
            # Hyperpriors
            mu_alpha = pm.Normal('mu_alpha', mu=0.75, sigma=0.1)
            sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=0.05)
            
            # Main effects
            alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha)
            
            # Random effects
            tau_model = pm.HalfNormal('tau_model', sigma=0.1)
            tau_domain = pm.HalfNormal('tau_domain', sigma=0.1)  
            tau_evaluator = pm.HalfNormal('tau_evaluator', sigma=0.05)
            
            model_effects = pm.Normal('model_effects', mu=0, sigma=tau_model, 
                                    shape=n_models)
            domain_effects = pm.Normal('domain_effects', mu=0, sigma=tau_domain,
                                     shape=n_domains)
            evaluator_effects = pm.Normal('evaluator_effects', mu=0, sigma=tau_evaluator,
                                        shape=n_evaluators)
            
            # Interaction effects
            tau_interaction = pm.HalfNormal('tau_interaction', sigma=0.05)
            interaction_effects = pm.Normal('interaction_effects', mu=0, 
                                          sigma=tau_interaction,
                                          shape=(n_models, n_domains))
            
            # Linear predictor
            mu = (alpha + 
                  model_effects[model_ids] +
                  domain_effects[domain_ids] + 
                  evaluator_effects[evaluator_ids] +
                  interaction_effects[model_ids, domain_ids])
            
            # Observation noise
            sigma_obs = pm.HalfNormal('sigma_obs', sigma=0.1)
            
            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs, 
                            observed=data['score'])
            
            self.model = hierarchical_model
        
        return hierarchical_model
    
    def run_inference(self, draws=4000, tune=2000, chains=4):
        """Run MCMC inference on the hierarchical model."""
        
        with self.model:
            # Use NUTS sampler for robust sampling
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=0.95,
                return_inferencedata=True
            )
        
        return self.trace
    
    def posterior_analysis(self):
        """Analyze posterior distributions and generate insights."""
        
        # Posterior summaries
        summary = pm.summary(self.trace)
        print("=== POSTERIOR SUMMARIES ===")
        print(summary)
        
        # Model comparison using LOO-CV
        loo = pm.loo(self.trace, self.model)
        print(f"\nLOO-CV Score: {loo.loo:.2f} ± {loo.loo_se:.2f}")
        
        # Posterior predictive checks
        with self.model:
            ppc = pm.sample_posterior_predictive(
                self.trace, samples=1000
            )
        
        return summary, loo, ppc
    
    def credible_intervals(self, parameter_name, credible_mass=0.95):
        """Calculate credible intervals for parameters."""
        
        posterior_samples = self.trace.posterior[parameter_name].values.flatten()
        
        alpha = 1 - credible_mass
        lower = np.percentile(posterior_samples, 100 * alpha/2)
        upper = np.percentile(posterior_samples, 100 * (1 - alpha/2))
        
        return lower, upper
```

### **3. Advanced Experimental Designs**

**Design 3.1** (Latin Square Design for Counterbalancing):
```python
class LatinSquareExperiment:
    """
    Latin square design for controlling order effects in evaluation.
    Ensures each model is tested on each domain in each position exactly once.
    """
    
    def __init__(self, models, domains):
        self.models = models
        self.domains = domains
        self.n = len(models)
        
        if len(domains) != self.n:
            raise ValueError("Number of models and domains must be equal for Latin Square")
    
    def generate_latin_square(self):
        """Generate balanced Latin square design."""
        
        # Start with a standard Latin square
        square = np.zeros((self.n, self.n), dtype=int)
        
        # Fill first row
        square[0] = np.arange(self.n)
        
        # Fill subsequent rows using cyclic permutation
        for i in range(1, self.n):
            square[i] = np.roll(square[i-1], 1)
        
        # Randomize rows and columns to avoid systematic bias
        row_perm = np.random.permutation(self.n)
        col_perm = np.random.permutation(self.n)
        
        square = square[row_perm][:, col_perm]
        
        return square
    
    def create_experiment_schedule(self):
        """Create experimental schedule from Latin square."""
        
        square = self.generate_latin_square()
        schedule = []
        
        for session in range(self.n):
            for position in range(self.n):
                model_idx = session
                domain_idx = square[session, position]
                
                schedule.append({
                    'session': session,
                    'position': position,
                    'model': self.models[model_idx],
                    'domain': self.domains[domain_idx]
                })
        
        return pd.DataFrame(schedule)
```

**Design 3.2** (Crossed Random Effects Design):
```python
class CrossedRandomEffectsDesign:
    """
    Crossed random effects design for comprehensive model evaluation.
    Accounts for random variation across multiple factors simultaneously.
    """
    
    def __init__(self, models, evaluators, test_sets, replications=3):
        self.models = models
        self.evaluators = evaluators  
        self.test_sets = test_sets
        self.replications = replications
        
    def generate_full_factorial_design(self):
        """Generate full factorial design matrix."""
        
        from itertools import product
        
        # Create full factorial combination
        combinations = list(product(
            range(len(self.models)),
            range(len(self.evaluators)),
            range(len(self.test_sets)),
            range(self.replications)
        ))
        
        # Randomize execution order
        np.random.shuffle(combinations)
        
        # Create design matrix
        design = []
        for model_idx, eval_idx, test_idx, rep in combinations:
            design.append({
                'model_id': model_idx,
                'model': self.models[model_idx],
                'evaluator_id': eval_idx,
                'evaluator': self.evaluators[eval_idx],
                'test_set_id': test_idx,
                'test_set': self.test_sets[test_idx],
                'replication': rep,
                'run_order': len(design) + 1
            })
        
        return pd.DataFrame(design)
    
    def analyze_variance_components(self, results):
        """Estimate variance components using REML."""
        
        from statsmodels.formula.api import mixedlm
        
        # Fit mixed effects model
        model = mixedlm(
            "score ~ 1", 
            results, 
            groups=results["model"],
            re_formula="1",
            vc_formula={
                "evaluator": "0 + C(evaluator)",
                "test_set": "0 + C(test_set)", 
                "interaction": "0 + C(model):C(evaluator)"
            }
        )
        
        fitted_model = model.fit()
        
        # Extract variance components
        variance_components = {
            'model_variance': fitted_model.cov_re,
            'evaluator_variance': fitted_model.vcomp[0],
            'test_set_variance': fitted_model.vcomp[1],
            'interaction_variance': fitted_model.vcomp[2],
            'residual_variance': fitted_model.scale
        }
        
        return fitted_model, variance_components
```

## 🔍 **Reliability and Validity Analysis**

### **1. Inter-Rater Reliability**

**Definition 4.1** (Intraclass Correlation Coefficient):
For evaluating consistency across multiple evaluators:

$$ICC(2,k) = \frac{MS_R - MS_E}{MS_R + (k-1)MS_E + \frac{k}{n}(MS_C - MS_E)}$$

where:
- $MS_R$: Mean square for rows (subjects)
- $MS_E$: Mean square for error  
- $MS_C$: Mean square for columns (raters)
- $k$: Number of raters
- $n$: Number of subjects

**Implementation**:
```python
import pandas as pd
import numpy as np
from scipy import stats

class ReliabilityAnalysis:
    """Comprehensive reliability analysis for evaluation systems."""
    
    def calculate_icc(self, ratings_matrix, icc_type='ICC(2,k)'):
        """
        Calculate Intraclass Correlation Coefficient.
        
        Args:
            ratings_matrix: n_subjects x n_raters matrix
            icc_type: Type of ICC to calculate
        """
        
        n_subjects, n_raters = ratings_matrix.shape
        
        # Calculate sum of squares
        grand_mean = np.mean(ratings_matrix)
        
        # Between subjects sum of squares
        subject_means = np.mean(ratings_matrix, axis=1)
        ss_between = n_raters * np.sum((subject_means - grand_mean)**2)
        
        # Within subjects sum of squares  
        ss_within = np.sum((ratings_matrix - subject_means.reshape(-1, 1))**2)
        
        # Between raters sum of squares
        rater_means = np.mean(ratings_matrix, axis=0)
        ss_raters = n_subjects * np.sum((rater_means - grand_mean)**2)
        
        # Error sum of squares
        ss_error = ss_within - ss_raters
        
        # Calculate mean squares
        ms_between = ss_between / (n_subjects - 1)
        ms_raters = ss_raters / (n_raters - 1)
        ms_error = ss_error / ((n_subjects - 1) * (n_raters - 1))
        
        # Calculate ICC based on type
        if icc_type == 'ICC(2,k)':
            # Two-way random effects, average measures
            icc = (ms_between - ms_error) / (ms_between + (n_raters - 1) * ms_error)
        elif icc_type == 'ICC(2,1)':
            # Two-way random effects, single measure
            icc = (ms_between - ms_error) / (ms_between + (n_raters - 1) * ms_error + 
                                           n_raters * (ms_raters - ms_error) / n_subjects)
        
        # Calculate confidence interval
        f_value = ms_between / ms_error
        df1 = n_subjects - 1
        df2 = n_subjects * (n_raters - 1)
        
        f_lower = f_value / stats.f.ppf(0.975, df1, df2)
        f_upper = f_value / stats.f.ppf(0.025, df1, df2)
        
        icc_lower = (f_lower - 1) / (f_lower + n_raters - 1)
        icc_upper = (f_upper - 1) / (f_upper + n_raters - 1)
        
        return {
            'ICC': icc,
            'confidence_interval': (icc_lower, icc_upper),
            'F_statistic': f_value,
            'p_value': 1 - stats.f.cdf(f_value, df1, df2)
        }
    
    def calculate_cronbach_alpha(self, ratings_matrix):
        """Calculate Cronbach's alpha for internal consistency."""
        
        n_subjects, n_items = ratings_matrix.shape
        
        # Calculate item variances
        item_variances = np.var(ratings_matrix, axis=0, ddof=1)
        
        # Calculate total variance
        total_variance = np.var(np.sum(ratings_matrix, axis=1), ddof=1)
        
        # Calculate Cronbach's alpha
        alpha = (n_items / (n_items - 1)) * (1 - np.sum(item_variances) / total_variance)
        
        return alpha
```

### **2. Construct Validity Analysis**

**Confirmatory Factor Analysis**:
```python
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
import numpy as np

class ConstructValidityAnalysis:
    """Analyze construct validity of evaluation dimensions."""
    
    def __init__(self):
        self.factor_model = None
        self.scaler = StandardScaler()
        
    def confirmatory_factor_analysis(self, dimension_scores, n_factors=4):
        """
        Perform confirmatory factor analysis on evaluation dimensions.
        
        Expected factors:
        1. Cognitive Quality (organization + accuracy)
        2. Content Quality (completeness + reliability)  
        3. Task-specific factors
        4. Method variance
        """
        
        # Standardize scores
        scores_standardized = self.scaler.fit_transform(dimension_scores)
        
        # Fit factor analysis model
        self.factor_model = FactorAnalysis(
            n_components=n_factors,
            rotation='varimax',
            random_state=42
        )
        
        factor_scores = self.factor_model.fit_transform(scores_standardized)
        
        # Calculate goodness of fit
        ll_null = self._calculate_null_likelihood(scores_standardized)
        ll_model = self.factor_model.score(scores_standardized)
        
        # Chi-square test statistic
        n_samples = scores_standardized.shape[0]
        chi_square = 2 * n_samples * (ll_model - ll_null)
        
        # Degrees of freedom
        n_vars = scores_standardized.shape[1]
        df = (n_vars * (n_vars - 1) // 2) - (n_vars * n_factors - n_factors * (n_factors - 1) // 2)
        
        # Calculate fit indices
        fit_indices = self._calculate_fit_indices(chi_square, df, n_samples, n_vars)
        
        return {
            'factor_loadings': self.factor_model.components_,
            'factor_scores': factor_scores,
            'fit_indices': fit_indices,
            'explained_variance': np.sum(self.factor_model.components_**2, axis=1)
        }
    
    def _calculate_fit_indices(self, chi_square, df, n_samples, n_vars):
        """Calculate common fit indices for factor analysis."""
        
        # Root Mean Square Error of Approximation (RMSEA)
        rmsea = np.sqrt(max(0, (chi_square - df) / (df * (n_samples - 1))))
        
        # Comparative Fit Index (CFI)  
        chi_square_null = n_vars * (n_vars - 1) / 2 * n_samples  # Approximate
        cfi = 1 - max(0, chi_square - df) / max(chi_square - df, chi_square_null - n_vars)
        
        # Tucker-Lewis Index (TLI)
        tli = (chi_square_null / (n_vars - 1) - chi_square / df) / (chi_square_null / (n_vars - 1) - 1)
        
        return {
            'chi_square': chi_square,
            'df': df,
            'p_value': 1 - stats.chi2.cdf(chi_square, df),
            'rmsea': rmsea,
            'cfi': cfi,
            'tli': tli
        }
```

### **3. Convergent and Discriminant Validity**

**Multitrait-Multimethod Analysis**:
```python
class MultitraitMultimethodAnalysis:
    """
    Analyze convergent and discriminant validity using MTMM matrix.
    Examines evaluation of same traits (dimensions) using different methods (evaluators).
    """
    
    def __init__(self):
        self.mtmm_matrix = None
        
    def create_mtmm_matrix(self, data):
        """
        Create multitrait-multimethod correlation matrix.
        
        Args:
            data: DataFrame with columns [trait, method, score]
        """
        
        # Pivot data to create correlation matrix structure
        pivot_data = data.pivot_table(
            index='subject_id', 
            columns=['trait', 'method'], 
            values='score'
        )
        
        # Calculate correlation matrix
        self.mtmm_matrix = pivot_data.corr()
        
        return self.mtmm_matrix
    
    def analyze_validity(self, traits, methods):
        """Analyze convergent and discriminant validity patterns."""
        
        results = {
            'convergent_validity': [],
            'discriminant_validity': [],
            'method_effects': []
        }
        
        # Analyze convergent validity (same trait, different methods)
        for trait in traits:
            trait_correlations = []
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods):
                    if i < j:  # Avoid duplicate pairs
                        corr = self.mtmm_matrix.loc[(trait, method1), (trait, method2)]
                        trait_correlations.append(corr)
            
            results['convergent_validity'].append({
                'trait': trait,
                'mean_correlation': np.mean(trait_correlations),
                'std_correlation': np.std(trait_correlations),
                'correlations': trait_correlations
            })
        
        # Analyze discriminant validity (different traits, same method)
        for method in methods:
            method_correlations = []
            for i, trait1 in enumerate(traits):
                for j, trait2 in enumerate(traits):
                    if i < j:  # Avoid duplicate pairs
                        corr = self.mtmm_matrix.loc[(trait1, method), (trait2, method)]
                        method_correlations.append(corr)
            
            results['discriminant_validity'].append({
                'method': method,
                'mean_correlation': np.mean(method_correlations),
                'std_correlation': np.std(method_correlations),
                'correlations': method_correlations
            })
        
        return results
    
    def campbell_fiske_criteria(self, mtmm_results):
        """Apply Campbell & Fiske (1959) criteria for validity assessment."""
        
        criteria_met = {
            'convergent_validity': True,
            'discriminant_validity_1': True,  # Same trait correlations > different trait correlations
            'discriminant_validity_2': True   # Same trait correlations > method correlations
        }
        
        # Check convergent validity (correlations should be high and significant)
        convergent_correlations = [
            result['mean_correlation'] 
            for result in mtmm_results['convergent_validity']
        ]
        
        if np.mean(convergent_correlations) < 0.7:
            criteria_met['convergent_validity'] = False
        
        # Check discriminant validity criteria
        discriminant_correlations = [
            result['mean_correlation']
            for result in mtmm_results['discriminant_validity']
        ]
        
        if np.mean(discriminant_correlations) >= np.mean(convergent_correlations):
            criteria_met['discriminant_validity_1'] = False
        
        return criteria_met
```

## 📈 **Longitudinal Analysis and Temporal Validity**

### **1. Time Series Analysis of Evaluation Stability**

```python
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

class TemporalValidityAnalysis:
    """Analyze temporal stability and trends in evaluation scores."""
    
    def __init__(self):
        self.time_series_models = {}
        
    def test_stationarity(self, score_series, alpha=0.05):
        """Test for stationarity using ADF and KPSS tests."""
        
        # Augmented Dickey-Fuller test (null: unit root/non-stationary)
        adf_result = adfuller(score_series)
        adf_statistic = adf_result[0]
        adf_p_value = adf_result[1]
        adf_critical_values = adf_result[4]
        
        # KPSS test (null: stationary)
        kpss_result = kpss(score_series)
        kpss_statistic = kpss_result[0]
        kpss_p_value = kpss_result[1]
        kpss_critical_values = kpss_result[3]
        
        # Interpret results
        is_stationary_adf = adf_p_value < alpha
        is_stationary_kpss = kpss_p_value > alpha
        
        return {
            'adf_test': {
                'statistic': adf_statistic,
                'p_value': adf_p_value,
                'critical_values': adf_critical_values,
                'is_stationary': is_stationary_adf
            },
            'kpss_test': {
                'statistic': kpss_statistic,
                'p_value': kpss_p_value,  
                'critical_values': kpss_critical_values,
                'is_stationary': is_stationary_kpss
            },
            'conclusion': is_stationary_adf and is_stationary_kpss
        }
    
    def fit_arima_model(self, score_series, order=(1, 0, 1)):
        """Fit ARIMA model to evaluation time series."""
        
        model = ARIMA(score_series, order=order)
        fitted_model = model.fit()
        
        # Model diagnostics
        residuals = fitted_model.resid
        ljung_box_stat = stats.diagnostic.acorr_ljungbox(residuals, lags=10)
        
        # Forecast next period
        forecast = fitted_model.forecast(steps=1)
        forecast_se = fitted_model.forecast(steps=1, alpha=0.05)[1]
        
        return {
            'model': fitted_model,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'ljung_box_p_value': ljung_box_stat.iloc[-1]['lb_pvalue'],
            'forecast': forecast[0],
            'forecast_se': forecast_se[0]
        }
    
    def detect_changepoints(self, score_series, method='PELT'):
        """Detect structural breaks/changepoints in evaluation scores."""
        
        try:
            import ruptures as rpt
            
            if method == 'PELT':
                # Pruned Exact Linear Time algorithm
                model = rpt.Pelt(model="rbf", min_size=10)
                changepoints = model.fit_predict(score_series.values, pen=1.0)
                
            elif method == 'Dynp':
                # Dynamic programming
                model = rpt.Dynp(model="l2", min_size=10)
                changepoints = model.fit_predict(score_series.values, n_bkps=3)
                
            # Remove last point (end of series)
            if changepoints[-1] == len(score_series):
                changepoints = changepoints[:-1]
            
            return {
                'changepoints': changepoints,
                'n_segments': len(changepoints) + 1,
                'segment_means': [
                    score_series.iloc[start:end].mean() 
                    for start, end in zip([0] + changepoints, changepoints + [len(score_series)])
                ]
            }
            
        except ImportError:
            # Fallback using simple variance-based detection
            return self._simple_changepoint_detection(score_series)
    
    def _simple_changepoint_detection(self, score_series, window_size=20):
        """Simple variance-based changepoint detection as fallback."""
        
        changepoints = []
        
        for i in range(window_size, len(score_series) - window_size):
            # Calculate variance before and after potential changepoint
            var_before = score_series.iloc[i-window_size:i].var()
            var_after = score_series.iloc[i:i+window_size].var()
            
            # F-test for variance equality
            f_statistic = max(var_before, var_after) / min(var_before, var_after)
            p_value = 2 * (1 - stats.f.cdf(f_statistic, window_size-1, window_size-1))
            
            if p_value < 0.05:  # Significant variance change
                changepoints.append(i)
        
        return {
            'changepoints': changepoints,
            'n_segments': len(changepoints) + 1
        }
```

## 🎯 **Meta-Analysis and Effect Size Estimation**

### **1. Random Effects Meta-Analysis**

```python
import numpy as np
from scipy import stats
import pandas as pd

class MetaAnalysisFramework:
    """
    Meta-analysis framework for aggregating evaluation results across studies.
    Uses random effects models to account for between-study heterogeneity.
    """
    
    def __init__(self):
        self.studies = []
        self.pooled_effect = None
        
    def add_study(self, effect_size, standard_error, study_name, sample_size):
        """Add study to meta-analysis."""
        
        # Calculate weight (inverse variance)
        variance = standard_error ** 2
        weight = 1 / variance
        
        self.studies.append({
            'study_name': study_name,
            'effect_size': effect_size,
            'standard_error': standard_error,
            'variance': variance,
            'weight': weight,
            'sample_size': sample_size
        })
    
    def calculate_pooled_effect(self):
        """Calculate pooled effect size using random effects model."""
        
        studies_df = pd.DataFrame(self.studies)
        
        # Fixed effects estimate (for heterogeneity calculation)
        fixed_weight_sum = studies_df['weight'].sum()
        fixed_effect = (studies_df['effect_size'] * studies_df['weight']).sum() / fixed_weight_sum
        
        # Calculate Q statistic for heterogeneity
        q_statistic = ((studies_df['effect_size'] - fixed_effect) ** 2 * studies_df['weight']).sum()
        df = len(self.studies) - 1
        
        # Calculate tau-squared (between-study variance)
        c = fixed_weight_sum - (studies_df['weight'] ** 2).sum() / fixed_weight_sum
        tau_squared = max(0, (q_statistic - df) / c)
        
        # Random effects weights
        studies_df['random_weight'] = 1 / (studies_df['variance'] + tau_squared)
        random_weight_sum = studies_df['random_weight'].sum()
        
        # Pooled effect size and standard error
        pooled_effect = (studies_df['effect_size'] * studies_df['random_weight']).sum() / random_weight_sum
        pooled_se = np.sqrt(1 / random_weight_sum)
        
        # Confidence interval
        z_critical = stats.norm.ppf(0.975)  # 95% CI
        ci_lower = pooled_effect - z_critical * pooled_se
        ci_upper = pooled_effect + z_critical * pooled_se
        
        # Heterogeneity statistics
        i_squared = max(0, (q_statistic - df) / q_statistic) * 100  # I² statistic
        h_squared = q_statistic / df if df > 0 else 0  # H² statistic
        
        self.pooled_effect = {
            'effect_size': pooled_effect,
            'standard_error': pooled_se,
            'confidence_interval': (ci_lower, ci_upper),
            'z_score': pooled_effect / pooled_se,
            'p_value': 2 * (1 - stats.norm.cdf(abs(pooled_effect / pooled_se))),
            'q_statistic': q_statistic,
            'q_p_value': 1 - stats.chi2.cdf(q_statistic, df),
            'i_squared': i_squared,
            'h_squared': h_squared,
            'tau_squared': tau_squared,
            'n_studies': len(self.studies)
        }
        
        return self.pooled_effect
    
    def publication_bias_tests(self):
        """Test for publication bias using multiple methods."""
        
        if len(self.studies) < 10:
            return {"warning": "Need at least 10 studies for reliable publication bias tests"}
        
        studies_df = pd.DataFrame(self.studies)
        
        # Egger's test (regression of standardized effect on precision)
        standardized_effect = studies_df['effect_size'] / studies_df['standard_error']
        precision = 1 / studies_df['standard_error']
        
        # Regression: standardized_effect ~ precision
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(precision, standardized_effect)
        
        egger_test = {
            'intercept': intercept,
            'p_value': p_value,
            'significant_bias': p_value < 0.05
        }
        
        # Begg's test (rank correlation between effect size and variance)
        from scipy.stats import kendalltau
        tau, begg_p_value = kendalltau(studies_df['effect_size'], studies_df['variance'])
        
        begg_test = {
            'tau': tau,
            'p_value': begg_p_value,
            'significant_bias': begg_p_value < 0.05
        }
        
        return {
            'egger_test': egger_test,
            'begg_test': begg_test
        }
```

---

This experimental validation framework provides the rigorous statistical foundation necessary for scientific validation of AI evaluation methods. The framework combines classical experimental design principles with modern statistical techniques to ensure robust, replicable, and valid evaluation outcomes.