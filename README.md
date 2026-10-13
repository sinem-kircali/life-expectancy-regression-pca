# What Drives Life Expectancy? OLS, PCA/PCR and Elastic Net on World Bank Indicators

This project explains and predicts life expectancy at birth for 160 countries from 29 World Bank health, economic and demographic indicators. It compares an interpretable stepwise OLS model with principal component regression and a cross-validated Elastic Net, then predicts life expectancy for three countries where it is missing.

Built for the *Introduction to Data Science* course (MSc Data Science & Marketing Analytics, Erasmus University Rotterdam).

**→ Full analysis with code, tables and figures: [life_expectancy.md](life_expectancy.md)**

## Key results
| Model | Test RMSE (years) | Test R² | Predictors |
|---|---|---|---|
| **Elastic Net** (α = 0.22) | **3.33** | 0.78 | 16 |
| PCR (5 components) | 3.62 | 0.74 | 5 |
| OLS (stepwise AIC) | 3.77 | 0.72 | 9 |
| OLS + interaction | 3.83 | 0.71 | 10 |

- **Drivers** (OLS, adjusted R² = 0.87):
  - life expectancy rises with government health spending, sanitation access and HepB3 immunisation;
  - it falls with fertility, tuberculosis mortality and hypertension.
- **No interaction:** spending works about equally well at every sanitation level (p = 0.80).
- **Structure:**
  - PC1, a public-health and development axis, explains 33% of predictor variance, with a bootstrap CI of [30%, 37%].
  - The first four components stand clearly above a permutation-test noise band.
- **Honest comparison:** the Elastic Net is the most accurate, but on 32 test countries a bootstrap CI for its RMSE advantage includes zero (OLS − ENet: 0.41 years [−0.14, 0.94]).

## Methods
- 80/20 train/test split made **before** imputation and scaling. Training means are used for imputation and for standardisation.
- Log transforms for skewed spending, income and population variables (OLS).
- Stepwise AIC selection, with VIF and residual diagnostics.
- PCA, with the number of components chosen by the Kaiser rule, the 70% variance threshold and a permutation test. PC1's variance share is bootstrapped.
- PCR with $k$ fixed by rule before the test set is used. Test RMSE for every $k$ is shown only as a sensitivity check.
- Elastic Net: 10-fold × 5 repeated CV over (α, λ), then a fine λ path with `cv.glmnet`.

## Notes on this version
This is a cleaned version of the original report code:
- removed duplicated chunks;
- replaced a hard-coded variance table with computed values;
- widened the λ grid, because the previous best value sat on the grid edge;
- dropped an adjusted R² on the test set, which mixes in-sample and out-of-sample ideas;
- added a bootstrap of the test-set differences between models.

## Structure
```
life_expectancy.Rmd     source (R Markdown)
life_expectancy.md      rendered analysis, viewable on GitHub
life_expectancy_files/  figures
data/                   place the three CSV files here (not included)
```
The data comes from the World Bank [Health, Nutrition and Population Statistics](https://databank.worldbank.org/source/health-nutrition-and-population-statistics) database. To re-run the analysis, place the files `health_nutrition_population.csv`, `life_expectancy.csv` and `new_countries.csv` in `data/`, then run `rmarkdown::render("life_expectancy.Rmd")`.

## Tech stack
R · tidyverse · MASS · car · pls · caret · glmnet · boot · corrplot · R Markdown
