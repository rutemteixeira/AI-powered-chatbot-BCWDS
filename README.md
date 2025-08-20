# Assistente Fidélia - AI-Powered Chatbot for Commercial Agents

Tested sequential machine learning algorithms on time series data to forecast sales by product group over a 10-month period.

## Project Overview

This solution was developed as part of the Business Cases with Data Science course in the MSc in Data Science and Advanced Analytics program at NOVA IMS, Lisbon. The course provides hands-on experience with real-world cases presented by established clients. This particular case was proposed by Siemens. <br>

The client’s business need was to develop an AI-driven sales forecasting model for Siemens' Smart Infrastructure Division in Germany to predict monthly sales (aggregated by product group) over a 10-month horizon. The solution integrated internal sales history with external macroeconomic indicators to move beyond manual, biased forecasting methods. The goal was to enhance accuracy, optimize resource allocation, and provide a scalable, data-driven framework for strategic decision-making.

## Repository structure
- `external_data.folder` - all macroeconomical indicators introduced from external data to enrich our existing correlations
- `BCWDS_forecasting_code.ipynb` - main notebook code with full pipeline and model assesment
- `BCWDS_forecasting_report.ipynb` - full written report in pdf

 
## Technical Approach

| Phase              |	          Techniques                                                                                                  |
|--------------------|---------------------------------------------------------------------------------------------------------------------- |
|Data Preparation    | Advanced missing value imputation (Polynomial Regression, proxy variables), datetime formatting, monthly aggregation  |
|Feature Engineering | Holiday count feature engineering, Exhaustive Lag Analysis, Incorporation of external data                            |
|Model Selection     | Ensemble Model: Neural Prophet (for seasonality/trends) and LSTM (for complex temporal dependencies)                  |
|Evaluation	         | Root Mean Squared Error (RMSE), Visual analysis (actual vs. predicted plots)                                          |



## Technology Stack

- **Programming Language**: Python
- **AI Framework**: Retrieval-Augmented Generation (RAG)
- **APIs**: Open AI API (GPT-4-o-mini)
- **UI Framework**: Streamlit
- **Data Processing**: Text preprocessing, tokenization
- **Vector Database**: FAISS structure for semantic search
  

## Key Challenges
- Little context provided by the client, lack of clarity in data forced us to assumptions
- Merging and cleaning disparate datasets with complex missing value patterns
- Identifying the optimal lag (time delay) for each macroeconomic indicator's effect on sales for each product group
- Building a robust model that can perform accurately and generalize well throught different product groups and sales patterns

 
## Results
- **~0.17 Average RMSE** - Closer to 0 = more accurate predictions


## Key Observations and Future Improvements
- Seasonality is confirmed and detected by data analysis


## Authors
- André Lourenço
- Emir Kamiloglu
- Manuel Andrade
- Rute Teixeira
- Victor Silva

### More details on https://rutemteixeira.github.io/generic/p6-sales-forecasting.html

