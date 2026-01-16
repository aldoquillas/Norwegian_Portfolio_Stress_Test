# Multi-Asset Portfolio Stress Testing under Tail Dependence  
*A Case Study with Norwegian Allocation*  

# Project Overview
This project analyses a 60/40 Norwegian portfolio through advanced stress-testing techniques.  
It integrates equities (S&P 500, MSCI EMU) and sovereign bonds (U.S. & German) using Student-t and historical simulation distributions and a t-Copula to capture tail dependence among equities and interest rates factors.  
Results quantify joint extreme losses via VaR and Expected Shortfall (ES) at 99.5 %.

# Methodology Summary
- **Equities:** Historical Simulation | Student-t | EWMA   
- **Fixed Income:** PCA (3 PCs ≈ 95 % variance) + Normal | Student-t| EWMA   
- **Integration:** Student-t Copula calibrated on Gaussianized scores  
- **Simulation:** 100 000 Monte Carlo runs to estimate portfolio loss distribution  

## Key Results
| Metric | Value |
|:--|:--|
| VaR (99.5 %) | -10.22 % |
| Expected Shortfall | -14.38 % |
| Copula ν | 17.47 |
| Best Equity Model | Student-t & Historical Simulation |
| Best Bond Model | Student-t + PCA |

## Investment Implications
- Tail risk intensifies during systemic stress; diversification benefits diminish.  
- Student-t Copula framework better captures co-movements in crises.  
- Useful for risk budgeting and stress-testing under Basel III / Solvency II frameworks.  

## Report
Full PDF report: `Norwegian Model Portfolio Stress Test.pdf`

**Author:** Aldo Quillas, CFA®, FRM  
**Contact:** [LinkedIn](https://www.linkedin.com/in/aldo-quillas-cfa-frm-5535b854/) | [Email](mailto:aldoquillasp@gmail.com)
