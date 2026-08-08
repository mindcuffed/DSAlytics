\# DSAlytics



AI-Powered DSA Performance Analytics \& Personalized Practice Recommendation System.



\## Overview



DSAlytics analyzes LeetCode submission history to identify performance patterns, detect weak DSA topics, estimate future failure risk using machine learning, and generate personalized practice recommendations.



The project also includes an interactive Streamlit dashboard for visualizing the analysis.



\## Features



\- Automatic LeetCode submission data collection

\- Submission performance analysis

\- Difficulty-wise performance analysis

\- Topic-level weakness detection

\- Historical and recent performance analysis

\- Machine learning-based failure prediction

\- Walk-forward model validation

\- Feature importance analysis

\- Personalized DSA practice recommendations

\- Interactive Streamlit dashboard



\## Tech Stack



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Streamlit

\- Plotly

\- Requests

\- LeetCode GraphQL API



\## Project Structure



```text

DSAlytics/

├── dashboard/

├── data/

├── models/

├── results/

├── analyze\_weaknesses.py

├── feature\_engineering.py

├── fetch\_leetcode\_data.py

├── fetch\_problem\_catalog.py

├── fetch\_problem\_metadata.py

├── prepare\_dataset.py

├── recommend\_next.py

├── recommend\_problems.py

├── train\_model.py

├── update\_dsalytics.py

└── validate\_model.py

