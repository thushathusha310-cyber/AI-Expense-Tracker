# AI Expense Tracker

An AI-powered personal expense tracking and financial analysis dashboard built with Python and Streamlit.

## Project Overview

AI Expense Tracker is a simple and interactive web application designed to help users record, monitor, and analyze their daily expenses.

The application provides a modern dashboard with spending statistics, category analysis, monthly trends, budget management, and AI-based expense prediction.

## Features

- Interactive financial dashboard
- Add new expenses
- View complete expense history
- Download expense data as CSV
- Monthly spending analysis
- Spending trend visualization
- Category-wise expense analysis
- Budget management
- AI-based next expense prediction
- AI-generated spending insights
- Simple and user-friendly interface

## AI Expense Prediction

The application uses a Machine Learning model to predict the next expected expense.

A Linear Regression model is trained using previous transaction amounts and transaction numbers.

### Prediction Process

1. Load recorded expense data
2. Validate expense amounts
3. Create transaction numbers
4. Train the Linear Regression model
5. Predict the next transaction amount
6. Display the prediction on the dashboard

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Linear Regression
- CSV
- Git
- GitHub

## Project Structure

```text
AI-Expense-Tracker/
|
|-- app.py
|-- ml_model.py
|-- expenses.csv
|-- .gitignore
|-- README.md
`-- venv/

## Live Demo

## Live Demo

[Open FinTrack AI](https://ai-expense-tracker-thusha.streamlit.app/)