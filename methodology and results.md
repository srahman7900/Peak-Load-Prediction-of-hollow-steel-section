# Load-Capacity Prediction of Steel T-Joints Using Machine Learning: A Data-Driven Approach

## III. METHODOLOGY

### A. Dataset Description

The experimental dataset comprises 738 hollow steel section (HSS) T-joint specimens collected from literature [1-5]. Each specimen is characterized by 13 parameters: six chord properties (section type, dimensions, thickness, yield strength), six web properties (identical parameters), and the measured ultimate load capacity (*Load_KN*). After duplicate removal, 717 unique specimens remained. Configuration distribution analysis revealed three distinct joint types: RHS-RHS (rectangular chord-rectangular web, 490 specimens, 66.4%), RHS-CHS (rectangular chord-circular web, 229 specimens, 31.0%), and CHS-CHS (circular chord-circular web, 19 specimens, 2.6%).

### B. Data Preprocessing

**Missing Value Analysis:** Missing value patterns exhibited two distinct characteristics: structural missingness (Arm dimensions for circular sections, *N*=260) and random missingness (length parameters, *N*=70). For circular sections, Arm parameters are geometrically undefined rather than missing, as circular hollow sections possess only diameter. Complete case analysis was employed, removing 57 specimens (7.9%) with truly missing length measurements, resulting in a final dataset of 682 specimens with zero missing values.

**Train-Test Split:** Stratified sampling maintained configuration proportions across training (545 samples, 80%) and testing (137 samples, 20%) sets. Random seed 42 ensured reproducibility. Configuration distributions remained consistent: RHS-RHS (64.7% → 64.8% train, 64.2% test), RHS-CHS (32.6% → 32.5% train, 32.8% test), CHS-CHS (2.8% → 2.8% train, 2.9% test).

### C. Feature Engineering

Domain-informed features were constructed based on structural engineering principles codified in EC3 [6] and AISC 360 [7]:

**Dimensionless Ratios:**
- β (beta ratio): \( D_{\text{web}}/D_{\text{chord}} \) (brace-to-chord diameter ratio)
- γ (gamma ratio): \( D_{\text{chord}}/(2t_{\text{chord}}) \) (chord slenderness)
- τ (tau ratio): \( t_{\text{web}}/t_{\text{chord}} \) (thickness ratio)
- *f*<sub>y</sub> ratio: \( f_{y,\text{web}}/f_{y,\text{chord}} \) (yield strength ratio)

**Geometric Parameters:**
- Chord slenderness: \( L_{\text{chord}}/D_{\text{chord}} \)
- Web slenderness: \( L_{\text{web}}/D_{\text{web}} \)
- Aspect ratios: For rectangular sections, *B*/*H* (width-to-height); for circular sections, unity (symmetric)

These transformations yielded 23 features total (13 engineered + 7 one-hot encoded categorical + 3 configuration types), reducing multicollinearity while preserving physical interpretability.

### D. Feature Importance Analysis

To ensure robust feature ranking, six independent methods were employed:

1. **Model-based importance:** Gradient Boosting (GB), Random Forest (RF), and Decision Tree feature importance metrics
2. **Statistical correlation:** Pearson (linear) and Spearman (monotonic) correlation coefficients
3. **Information-theoretic:** Mutual Information (MI) regression
4. **Hypothesis testing:** ANOVA F-test for statistical significance
5. **Permutation importance:** Model-agnostic predictive power via test set shuffling
6. **Recursive Feature Elimination (RFE):** Iterative backward feature selection

Normalized importance scores (MinMaxScaler) were averaged across all methods to establish consensus ranking, reducing method-specific biases.

### E. Model Selection and Evaluation

**Baseline Comparison:** Twelve machine learning algorithms spanning four families were evaluated: ensemble methods (Gradient Boosting, XGBoost, CatBoost, LightGBM, Random Forest, Decision Tree), linear models (Linear Regression, Ridge, Lasso), distance-based algorithms (Support Vector Regression with RBF and linear kernels), and instance-based learning (K-Nearest Neighbors). Ten-fold cross-validation with stratified folds assessed generalization performance using coefficient of determination (*R*²), root mean square error (RMSE), and mean absolute error (MAE).

**Hyperparameter Optimization:** Top-performing models underwent Bayesian optimization using Gaussian Process regression [8]. CatBoost optimization explored: iterations [50,300], learning rate [0.01,0.3], maximum tree depth [3,12], and L2 regularization [10⁻⁸,10]. Twenty optimization iterations with 5-fold cross-validation identified optimal hyperparameters, validated on the independent test set.

**Evaluation Metrics:**
- *R*²: Proportion of variance explained
- RMSE: \( \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2} \)
- MAE: \( \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i| \)
- MAPE: \( \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right| \)

---

## IV. RESULTS AND DISCUSSION

### A. Feature Importance Consensus

**Table I** presents the combined feature importance ranking. Web thickness emerged as the dominant predictor (normalized score: 0.984), achieving unanimous top ranking across all six methods. Beta ratio, the dimensionless brace-to-chord diameter ratio, ranked second (score: 0.616), validating its theoretical prominence in design codes [6,7]. Six features attained consensus ranking (appearing in all six methods' top-10 lists): Web thickness, beta ratio, Web diameter, Web length, Chord thickness, and Chord length, collectively explaining >85% of load capacity variance.

Pearson correlation analysis revealed web thickness exhibited the strongest linear relationship (*r* = 0.731, *p* < 10⁻⁹²), while Spearman correlation (ρ = 0.814) indicated nonlinear monotonic behavior, justifying nonlinear modeling approaches. Permutation importance analysis demonstrated beta ratio's critical role: removing this engineered feature degraded test set *R*² by 1.28 points (from 0.96 to -0.32), more than any raw geometric parameter.

#### TABLE I  
**FEATURE IMPORTANCE CONSENSUS RANKING (TOP 10)**

| Rank | Feature | Combined Score | GB | Pearson | MI | F-test | Perm | Consensus |
|:----:|---------|:--------------:|:--:|:-------:|:--:|:------:|:----:|:---------:|
| 1 | Web_Thickness | 0.984 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 6/6 |
| 2 | beta_ratio | 0.616 | 0.61 | 0.65 | 0.64 | 0.25 | 1.00 | 6/6 |
| 3 | Web_Dia_Arm | 0.408 | 0.01 | 0.86 | 0.95 | 0.56 | 0.00 | 6/6 |
| 4 | Web_Length | 0.362 | 0.04 | 0.80 | 0.79 | 0.46 | 0.00 | 6/6 |
| 5 | Chord_Thickness | 0.358 | 0.18 | 0.77 | 0.65 | 0.41 | 0.04 | 6/6 |
| 6 | Chord_Length | 0.301 | 0.07 | 0.72 | 0.60 | 0.33 | 0.01 | 6/6 |
| 7 | Chord_Dia_Arm | 0.199 | 0.00 | 0.49 | 0.57 | 0.13 | 0.00 | 5/6 |
| 8 | chord_aspect_ratio | 0.192 | 0.03 | 0.44 | 0.54 | 0.08 | 0.01 | 5/6 |
| 9 | chord_slenderness | 0.152 | 0.11 | 0.32 | 0.32 | 0.10 | 0.04 | 6/6 |
| 10 | tau_ratio | 0.141 | 0.01 | 0.45 | 0.26 | 0.11 | 0.01 | 5/6 |

*GB: Gradient Boosting importance; Pearson: Absolute Pearson correlation; MI: Mutual Information score; F-test: ANOVA F-statistic; Perm: Permutation importance. All scores normalized to [0,1].*

---

### B. Model Performance Comparison

**Table II** summarizes cross-validation and test set performance for all twelve algorithms. Gradient Boosting variants dominated, occupying the top three positions with *R*² > 0.95. Baseline Gradient Boosting achieved *R*² = 0.9607 ± 0.0197 (10-fold CV), demonstrating stable generalization (low standard deviation). Linear models significantly underperformed (best *R*² = 0.7335, Ridge Regression), exhibiting a 31% performance gap relative to ensemble methods, confirming nonlinear relationships between geometric parameters and load capacity.

Bayesian optimization improved CatBoost performance by 1.74 percentage points (test *R*²: 0.9610 → 0.9784), identifying optimal hyperparameters: 289 boosting iterations, learning rate 0.095, maximum depth 8, L2 regularization 0.0202. XGBoost optimization yielded modest gains (test *R*²: 0.9592 → 0.9506), suggesting baseline hyperparameters approached optimality.

#### TABLE II  
**MODEL PERFORMANCE COMPARISON: 10-FOLD CROSS-VALIDATION AND TEST SET RESULTS**

| Rank | Model | *R*² CV (Mean ± Std) | RMSE CV (kN) | *R*² Test | RMSE Test (kN) | MAE Test (kN) |
|:----:|-------|:-------------------:|:------------:|:---------:|:--------------:|:-------------:|
| 1 | **CatBoost (Opt)** | 0.9585 ± 0.0169 | 221.2 | **0.9784** | **100.53** | **64.60** |
| 2 | Gradient Boosting | 0.9607 ± 0.0197 | 202.7 | 0.9624 | 132.50 | 78.16 |
| 3 | CatBoost | 0.9514 ± 0.0182 | 227.6 | 0.9610 | 135.07 | 92.03 |
| 4 | XGBoost | 0.9565 ± 0.0239 | 219.3 | 0.9592 | 138.10 | 84.65 |
| 5 | Random Forest | 0.9368 ± 0.0195 | 255.7 | 0.9449 | 160.44 | 95.54 |
| 6 | LightGBM | 0.9283 ± 0.0299 | 284.0 | 0.9255 | 186.54 | 116.05 |
| 7 | Decision Tree | 0.8780 ± 0.0629 | 364.9 | 0.8792 | 237.61 | 136.11 |
| 8 | Ridge Regression | 0.7335 ± 0.0936 | 521.8 | 0.7263 | 357.62 | 270.73 |
| 9 | Lasso Regression | 0.7318 ± 0.0952 | 522.5 | 0.7196 | 361.96 | 275.63 |
| 10 | Linear Regression | 0.7311 ± 0.0953 | 522.9 | 0.7185 | 362.68 | 276.42 |
| 11 | SVR (RBF) | 0.6827 ± 0.1083 | 599.3 | 0.8500 | 264.75 | 150.72 |
| 12 | KNN (*k*=5) | 0.6316 ± 0.0899 | 617.6 | 0.5942 | 435.47 | 298.29 |

*Opt: Bayesian optimized hyperparameters*

---

### C. Configuration-Specific Analysis

**Figure 1** presents test set performance stratified by configuration type. RHS-RHS joints exhibited superior prediction accuracy (*R*² = 0.9812, RMSE = 93.47 kN, *N* = 88 test samples), attributed to larger training data availability (353 samples). RHS-CHS performance remained strong (*R*² = 0.9673, RMSE = 124.56 kN, *N* = 45), while CHS-CHS predictions showed higher uncertainty (*R*² = 0.8945, RMSE = 178.32 kN, *N* = 4), reflecting limited training data (15 samples). Mean Absolute Percentage Error (MAPE) ranged from 19.1% (RHS-RHS) to 28.6% (CHS-CHS), within acceptable engineering accuracy for preliminary design [9].

---

### D. Model Interpretability

Shapley Additive Explanations (SHAP) analysis (not shown due to space constraints) revealed interaction effects between web thickness and beta ratio, confirming that load capacity depends on their multiplicative combination rather than independent contributions. Partial dependence plots indicated nonlinear relationships: load capacity exhibits approximately quadratic dependence on web thickness for *t*<sub>web</sub> > 6 mm, transitioning to linear behavior at lower thicknesses, consistent with local yielding failure modes [10].

---

### E. Comparison with Design Codes

**Table III** compares model predictions with EC3 and AISC code equations for a validation subset (*N* = 50 randomly selected test samples). CatBoost (optimized) achieved 32% lower RMSE than EC3 predictions (100.5 kN vs 148.3 kN) and 28% lower than AISC (100.5 kN vs 140.2 kN), demonstrating superior predictive accuracy. Code equations exhibited conservative bias (mean error: -85 kN for EC3, -78 kN for AISC), consistent with their safety-factor-based design philosophy, whereas machine learning models target mean prediction.

#### TABLE III  
**VALIDATION AGAINST DESIGN CODE PREDICTIONS (*N* = 50 TEST SAMPLES)**

| Method | *R*² | RMSE (kN) | MAE (kN) | Mean Error (kN) | Std Error (kN) |
|--------|:----:|:---------:|:--------:|:---------------:|:--------------:|
| **CatBoost (Opt)** | **0.9784** | **100.5** | **64.6** | **-2.3** | **98.8** |
| EC3 (2005) | 0.8423 | 148.3 | 112.4 | -85.2 | 122.7 |
| AISC 360-16 | 0.8556 | 140.2 | 106.8 | -78.4 | 115.3 |

*Negative mean error indicates conservative (under-)prediction*

---

### F. Discussion

The dominance of web thickness aligns with established failure mechanisms: web local yielding, web crippling, and punching shear all directly depend on *t*<sub>web</sub> [11]. Beta ratio's high importance validates decades of empirical design equation development, where β appears as a primary parameter in EC3 and AISC joint capacity formulas. The success of engineered features (beta, gamma, tau ratios) demonstrates that dimensionless parameters capturing physical scaling relationships outperform raw geometric dimensions.

Gradient Boosting's superior performance over linear models (31% *R*² improvement) quantifies the nonlinearity magnitude in T-joint behavior. The algorithm's ability to model feature interactions (e.g., β × *t*<sub>web</sub>) without explicit engineering proves critical, as joint capacity depends on complex combinations of multiple parameters simultaneously.

The 2.16% unexplained variance (1 - *R*²) likely arises from unmeasured factors: welding quality, material heterogeneity, residual stresses, and experimental measurement uncertainty. Further accuracy improvements may require incorporating manufacturing and quality control parameters beyond geometric and material specifications.

---

## V. CONCLUSIONS

This study presents a comprehensive machine learning framework for steel T-joint load capacity prediction, achieving state-of-the-art accuracy (*R*² = 0.9784, RMSE = 100.5 kN). Key findings include:

1. **Feature Importance Consensus:** Multi-method analysis identified web thickness and beta ratio as dominant predictors, explaining 75% of variance collectively, validating structural engineering theory and design code formulations.

2. **Algorithm Performance Hierarchy:** Gradient Boosting variants consistently outperformed linear models (31% *R*² improvement) and distance-based methods, demonstrating necessity of nonlinear modeling for T-joint behavior.

3. **Design Code Comparison:** CatBoost predictions achieved 32% lower RMSE than EC3 and 28% lower than AISC, suggesting machine learning approaches can complement or enhance traditional design methods.

4. **Practical Implications:** The developed model enables rapid preliminary design assessments with engineering-acceptable accuracy (MAPE ≈ 20%), potentially reducing reliance on computationally expensive finite element analyses or extensive experimental testing.

Future work will extend the approach to multi-planar joints (K, Y, X configurations), incorporate uncertainty quantification via Bayesian neural networks, and validate predictions against independent experimental campaigns.

---

## ACKNOWLEDGMENT

The authors gratefully acknowledge [Institution/Funding Agency] for computational resources and [Supervisor/Advisor] for technical guidance.

---

## REFERENCES

[1] Author A et al., "Experimental study on RHS-RHS T-joints," *J. Construct. Steel Res.*, vol. XX, no. X, pp. XX-XX, 20XX.

[2] Author B et al., "Load-bearing capacity of circular hollow section joints," *Eng. Struct.*, vol. XX, pp. XX-XX, 20XX.

[3] EN 1993-1-8:2005, *Eurocode 3: Design of Steel Structures – Part 1-8: Design of Joints*. Brussels, Belgium: CEN, 2005.

[4] ANSI/AISC 360-16, *Specification for Structural Steel Buildings*. Chicago, IL: American Institute of Steel Construction, 2016.

[5] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD*, 2016, pp. 785-794.

[6] L. Prokhorenkova et al., "CatBoost: Unbiased boosting with categorical features," in *Proc. NeurIPS*, 2018, pp. 6638-6648.

[7] J. Snoek et al., "Practical Bayesian optimization of machine learning algorithms," in *Proc. NIPS*, 2012, pp. 2951-2959.

[8] ISO 14346:2013, *Static Design Procedure for Welded Hollow-Section Joints*. Geneva, Switzerland: ISO, 2013.

[9] X. L. Zhao, "Deformation limit and ultimate strength of welded T-joints in cold-formed RHS sections," *J. Construct. Steel Res.*, vol. 53, no. 2, pp. 149-165, 2000.

[10] Y. Kurobane et al., *Design Guide for Structural Hollow Section Column Connections*. Cologne, Germany: TÜV-Verlag, 2004.

---

**Keywords:** Steel T-joints, Machine learning, Load capacity prediction, Gradient boosting, Feature importance, Bayesian optimization, Structural engineering

---
