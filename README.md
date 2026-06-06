# Restaurant Order Prediction using Machine Learning
> **Forecasting Zomato Order Volumes to Optimize Staffing and Operations**

This repository contains a Jupyter Notebook implementation for predicting restaurant order volumes using historical order logs. The project performs exploratory data analysis, cleans data, engineers temporal features, and fits a supervised machine learning regression model to predict demand patterns.

---

## Project Overview

A major challenge for restaurants listed on food delivery platforms like Zomato is managing inventory and staffing resources. Over-staffing or over-stocking leads to food wastage and higher overheads, while under-staffing leads to delayed orders and poor customer reviews.

This project addresses this by:
- **Analyzing Order History**: Visualizing customer ordering behavior across top restaurants, subzones, and promotions.
- **Predicting Future Demand**: Implementing a `LinearRegression` model trained on chronological features to forecast daily order volume.
- **Generating Future Forecasts**: Predicting order volumes for a 7-day future horizon.

---

## Notebook Workflow

The analysis is structured inside [`oredr_prediction.ipynb`](oredr_prediction.ipynb) as follows:

1. **Libraries & Data Loading**:
   - Imports key tools: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, and `plotly`.
   - Reads the raw transaction dataset `order_history.csv`.
2. **Exploratory Data Analysis (EDA)**:
   - Evaluates missing values and prints descriptive statistics.
   - Plots interactive distribution charts (using `plotly.express`) for order volume across different **Subzones** and **Restaurant names**.
   - Analyzes the impact of promotional discounts (`Discount_Applied`) on order frequency.
3. **Data Encoding & Processing**:
   - Encodes categorical columns (Restaurant Name, Subzone) using `LabelEncoder`.
4. **Feature Engineering**:
   - Parses order dates to extract temporal features: `day`, `month`, and `year`.
   - Groups transactions to compute total daily orders.
5. **Model Training & Evaluation**:
   - Splits data chronologically (80% train, 20% test).
   - Trains a `LinearRegression` model.
   - Evaluates performance using standard regression metrics:
     - **Mean Absolute Error (MAE)**: `38.16` orders
     - **Mean Squared Error (MSE)**: `2356.8`
6. **Future Forecasting**:
   - Generates predictions for a 7-day future horizon.

---

## Installation & Usage

### 1. Install Dependencies
Make sure you have Python installed, then install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook
```
Open [`oredr_prediction.ipynb`](oredr_prediction.ipynb), ensure your dataset `order_history.csv` is present in the root folder, and run all cells.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
