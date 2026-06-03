# 🚗 Actuarial Auto Insurance Pricing Engine: GLM vs. XGBoost

[![Live Streamlit App](https://img.shields.io/badge/Live_App-Click_Here-blue.svg)](https://actuarial-pricing-engine-vjrpcku5ukswsgdt2dvo9z.streamlit.app/)

## 1. Executive Summary
Developed an end-to-end machine learning pricing engine to predict auto insurance claims frequency and severity. By migrating from a traditional Maximum Likelihood Gamma/Poisson GLM to a gradient-boosted tree architecture (XGBoost), this project identified non-linear risk interactions, resulting in a **+2.86% absolute lift in the Gini Index**. The model is deployed as an interactive Streamlit web application featuring real-time SHAP explainability.

## 2. The Business Problem
Traditional Generalized Linear Models (GLMs) are the industry standard for pricing, but they struggle to capture complex, non-linear interactions between driver features (e.g., the compounding risk of a young driver in a high-powered vehicle with a poor claims history). This leaves "money on the table" by underpricing risky profiles and overpricing safe profiles. 

## 3. The Technical Solution
* **Frequency Model:** XGBoost with Poisson regression, utilizing `log(Exposure)` as the base margin to calculate accurate term-expected frequency.
* **Severity Model:** XGBoost with Gamma regression, initialized with the portfolio's average claim cost to solve gradient boosting convergence limits on extreme tail risks.
* **Evaluation:** Both models were evaluated using Actuarial Double Lift Charts and Lorenz Curves to measure portfolio sorting efficiency.

## 4. Financial Impact & Results
The XGBoost challenger model outperformed the traditional GLM baseline significantly:
* **GLM Baseline Gini Index:** 34.02%
* **XGBoost Gini Index:** 36.88%
* **Net Lift:** +2.86% 

*<img width="854" height="552" alt="actuarial lift chart" src="https://github.com/user-attachments/assets/587609bf-cc04-4b72-9edd-0f2f141c2528" />*

*<img width="861" height="629" alt="actuarial lorenz curves" src="https://github.com/user-attachments/assets/21e08c7f-0d82-44ee-bb5c-dae20fe61ecd" />*


## 5. Regulatory Compliance & Explainable AI (XAI)
Machine Learning models are often rejected by insurance regulators (e.g., MAS, FCA) due to their "black-box" nature. To ensure regulatory compliance and assist underwriters, this project integrates **SHAP (SHapley Additive exPlanations)**. The web app dynamically generates a SHAP waterfall chart for every quote, mathematically proving exactly which features drove the premium up or down.

*<img width="881" height="744" alt="Screenshot 2026-06-03 at 7 32 12 PM" src="https://github.com/user-attachments/assets/3b12f7f1-df1c-4404-ac8d-19b1e897a8d1" />*

Screenshot of the app
