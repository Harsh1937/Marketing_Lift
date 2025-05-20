**Uplift Modeling
Project: Uplift Modeling Case Study – Identifying Persuadable Customers for a Marketing Campaign**

**Problem Statement** 
In many business scenarios, especially marketing, it is not enough to predict who is likely to purchase a product – we need to know who is likely to purchase because of a given intervention. Uplift modeling addresses this problem by estimating the causal effect of a treatment (such as a coupon, advertisement, or promotion) on an outcome uplift-modeling.com . The challenge is to distinguish between customers who would buy anyway, those who would never buy even with incentives, and those who will buy only if they receive the incentive. Traditional predictive models (like response classifiers) cannot easily differentiate these cases, but uplift models can. This project focuses on building an uplift model that can find persuadable customers – those who are likely to make a purchase only when treated with a marketing action uplift-modeling.com . By targeting such customers, businesses can maximize campaign ROI and avoid wasting resources on customers who would respond regardless or not at all.

**Dataset** 
The project uses the Online Retail dataset (from the UCI Machine Learning Repository and also available on Kaggle). This is a real transactional dataset containing all the e-commerce transactions for a UK-based online retail company from December 2010 to December 2011 archive.ics.uci.edu . The data consists of 541,909 transactions (rows) involving 4,372 unique customers, with the following features for each transaction archive.ics.uci.edu archive.ics.uci.edu : InvoiceNo – Unique identifier for each transaction (invoice). StockCode – Product (item) code for the item sold. Description – Text description of the product. Quantity – Number of units of the product sold in that transaction line. InvoiceDate – Date and time of the transaction. UnitPrice – Unit price of the product (in GBP). CustomerID – Unique ID of the customer who made the purchase. Country – Country of residence of the customer. Data characteristics: The dataset spans purchases over a one-year period and includes retail transactions for a variety of gift items archive.ics.uci.edu . It is transactional (multiple rows per customer) and may include some irregularities such as canceled orders (denoted by InvoiceNo starting with "C") or missing customer IDs for some entries. For the purposes of this project, the data was cleaned to remove anomalies (e.g., negative quantities indicating returns, transactions without a CustomerID, etc.), focusing on genuine completed purchases. Aggregating to Customer-Level: Since uplift modeling in this project is done at the customer level (to decide which customers to target), the transactional data was aggregated by CustomerID. This means we derived features that summarize each customer’s historical purchase behavior from the raw transactions. Each customer becomes one data point in the modeling dataset, with aggregated features described below.

**Feature Engineering**
To prepare the dataset for uplift modeling, the following steps were applied:

TotalPrice: Calculated as Quantity * UnitPrice to capture revenue per transaction. Temporal Feature: Extracted Hour from InvoiceDate to capture time-of-day purchasing behavior. Customer Aggregation: Grouped by CustomerID to create customer-level features: TotalSpend: Total revenue from a customer NumPurchases: Number of invoices AvgHour: Average purchase hour Simulated Treatment: Assigned a random binary treatment flag (50/50 split) to emulate an A/B test setup. Simulated Target: Defined target = 1 if TotalPrice > 20 to simulate a positive purchase outcome. Treated customers were given a slightly higher likelihood of response than control, modeling a realistic uplift effect.

After processing, each customer is represented by a single row with aggregated features, a treatment flag, and a binary target — ready for uplift modeling.

**Evaluation Metrics**
Standard classification metrics don't fully capture uplift performance, so we use uplift-specific metrics:

Uplift AUC (AUUC): Measures the area under the uplift curve, which tracks the incremental gain in positive outcomes as we target more customers based on predicted uplift. A score of 1.0 is perfect; 0.5 equals random targeting.

Qini AUC: A normalized version of the uplift curve that compares the model to both a perfect and random baseline. It shows how well the model prioritizes persuadable customers. Scores typically range between 0.1–0.3 in practice, where higher is better.

These metrics were calculated using scikit-uplift's uplift_auc_score and qini_auc_score. Visual curves and additional metrics like Uplift@K are also available, but AUUC and Qini AUC are the focus here.

**Author Created by Harsh Shah**
