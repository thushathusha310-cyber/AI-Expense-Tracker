import streamlit as st
import pandas as pd
import os
from datetime import date

from ml_model import predict_next_expense


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="FinTrack AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# FILE SETUP
# =========================================================

FILE_NAME = "expenses.csv"

if not os.path.exists(FILE_NAME):
    empty_df = pd.DataFrame(
        columns=[
            "Date",
            "Category",
            "Amount",
            "Description"
        ]
    )

    empty_df.to_csv(
        FILE_NAME,
        index=False
    )


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    try:
        df = pd.read_csv(FILE_NAME)

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "Date",
                    "Category",
                    "Amount",
                    "Description"
                ]
            )

        df["Amount"] = pd.to_numeric(
            df["Amount"],
            errors="coerce"
        )

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["Amount"]
        )

        return df

    except Exception:
        return pd.DataFrame(
            columns=[
                "Date",
                "Category",
                "Amount",
                "Description"
            ]
        )


df = load_data()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ================================
       MAIN PAGE
       ================================ */

    .stApp {
        background-color: #f5f7fb;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }


    /* ================================
       SIDEBAR
       ================================ */

    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: none;
    }

    section[data-testid="stSidebar"] > div {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff;
    }

    .sidebar-brand {
        font-size: 25px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .sidebar-subtitle {
        color: #9ca3af !important;
        font-size: 13px;
        margin-bottom: 30px;
    }

    .sidebar-menu-title {
        color: #6b7280 !important;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
    }


    /* ================================
       PAGE HEADER
       ================================ */

    .page-title {
        font-size: 32px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 2px;
    }

    .page-subtitle {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 25px;
    }


    /* ================================
       WELCOME CARD
       ================================ */

    .welcome-card {
        background: linear-gradient(
            135deg,
            #172554,
            #312e81
        );
        border-radius: 22px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(31, 41, 55, 0.12);
    }

    .welcome-small {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.4px;
        opacity: 0.75;
        margin-bottom: 8px;
    }

    .welcome-title {
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .welcome-text {
        font-size: 14px;
        line-height: 1.6;
        opacity: 0.85;
        max-width: 700px;
    }


    /* ================================
       METRIC CARDS
       ================================ */

    .metric-card {
        background: white;
        border-radius: 18px;
        padding: 22px;
        min-height: 145px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    }

    .metric-icon {
        font-size: 25px;
        margin-bottom: 12px;
    }

    .metric-title {
        color: #6b7280;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #111827;
        font-size: 25px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .metric-description {
        color: #9ca3af;
        font-size: 12px;
    }


    /* ================================
       SECTION TITLE
       ================================ */

    .section-title {
        color: #111827;
        font-size: 20px;
        font-weight: 800;
        margin-top: 30px;
        margin-bottom: 3px;
    }

    .section-subtitle {
        color: #6b7280;
        font-size: 13px;
        margin-bottom: 15px;
    }


    /* ================================
       CARDS
       ================================ */

    .content-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
    }


    /* ================================
       TRANSACTION CARD
       ================================ */

    .transaction-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 15px 18px;
        margin-bottom: 10px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);
    }

    .transaction-name {
        font-size: 14px;
        font-weight: 700;
        color: #111827;
    }

    .transaction-info {
        font-size: 12px;
        color: #6b7280;
        margin-top: 3px;
    }

    .transaction-amount {
        font-size: 15px;
        font-weight: 800;
        color: #dc2626;
        text-align: right;
    }


    /* ================================
       AI CARD
       ================================ */

    .ai-card {
        background: linear-gradient(
            135deg,
            #312e81,
            #4f46e5
        );
        border-radius: 20px;
        padding: 24px;
        color: white;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.18);
    }

    .ai-label {
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.5px;
        opacity: 0.75;
    }

    .ai-value {
        font-size: 32px;
        font-weight: 800;
        margin-top: 8px;
    }

    .ai-description {
        font-size: 13px;
        opacity: 0.8;
        margin-top: 4px;
    }


    /* ================================
       BUDGET CARD
       ================================ */

    .budget-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
    }

    .budget-title {
        color: #111827;
        font-size: 17px;
        font-weight: 800;
    }

    .budget-value {
        color: #111827;
        font-size: 28px;
        font-weight: 800;
        margin-top: 8px;
    }

    .budget-description {
        color: #6b7280;
        font-size: 13px;
        margin-bottom: 12px;
    }


    /* ================================
       BUTTONS
       ================================ */

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        border: 1px solid #d1d5db;
        min-height: 42px;
    }

    .stButton > button:hover {
        border-color: #4f46e5;
        color: #4f46e5;
    }


    /* ================================
       INPUTS
       ================================ */

    div[data-baseweb="input"] {
        border-radius: 10px;
    }

    div[data-baseweb="select"] {
        border-radius: 10px;
    }


    /* ================================
       FOOTER
       ================================ */

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 12px;
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid #e5e7eb;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">💰 FinTrack AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">Smart Personal Finance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-menu-title">MAIN MENU</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "",
        [
            "🏠 Dashboard",
            "➕ Add Expense",
            "📜 Expense History",
            "📊 Monthly Analysis",
            "🤖 AI Insights",
            "💰 Budget Management"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.caption("FinTrack AI")
    st.caption("Smart Personal Finance Dashboard")


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="page-title">Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">Your personal finance dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="welcome-card">
            <div class="welcome-small">WELCOME BACK 👋</div>
            <div class="welcome-title">Manage your money smarter.</div>
            <div class="welcome-text">
                Track expenses, understand spending patterns,
                and use AI to plan your finances.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------
    # Calculations
    # -----------------------------

    if not df.empty:

        total_spending = df["Amount"].sum()
        transactions = len(df)
        average_expense = df["Amount"].mean()
        highest_expense = df["Amount"].max()

    else:

        total_spending = 0
        transactions = 0
        average_expense = 0
        highest_expense = 0


    # -----------------------------
    # Metric Cards
    # -----------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">💳</div>
                <div class="metric-title">TOTAL SPENDING</div>
                <div class="metric-value">
                    Rs. {total_spending:,.0f}
                </div>
                <div class="metric-description">
                    All recorded expenses
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">🧾</div>
                <div class="metric-title">TRANSACTIONS</div>
                <div class="metric-value">
                    {transactions}
                </div>
                <div class="metric-description">
                    Total transactions
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">📈</div>
                <div class="metric-title">AVERAGE EXPENSE</div>
                <div class="metric-value">
                    Rs. {average_expense:,.0f}
                </div>
                <div class="metric-description">
                    Per transaction
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">🔥</div>
                <div class="metric-title">HIGHEST EXPENSE</div>
                <div class="metric-value">
                    Rs. {highest_expense:,.0f}
                </div>
                <div class="metric-description">
                    Largest transaction
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # -----------------------------
    # Spending Trend
    # -----------------------------

    st.markdown(
        '<div class="section-title">Spending Trend</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Your expense activity</div>',
        unsafe_allow_html=True
    )

    if not df.empty:

        trend_df = df.copy()

        trend_df["Date"] = pd.to_datetime(
            trend_df["Date"],
            errors="coerce"
        )

        trend_df = trend_df.dropna(
            subset=["Date"]
        )

        daily_spending = (
            trend_df
            .groupby("Date")["Amount"]
            .sum()
            .sort_index()
        )

        if not daily_spending.empty:

            st.line_chart(
                daily_spending,
                height=320
            )

        else:

            st.info("No valid dates available.")

    else:

        st.info("Add some expenses to see your spending trend.")


    # -----------------------------
    # Category Chart
    # -----------------------------

    st.markdown(
        '<div class="section-title">Spending by Category</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Category distribution</div>',
        unsafe_allow_html=True
    )

    if not df.empty:

        category_data = (
            df.groupby("Category")["Amount"]
            .sum()
            .sort_values(ascending=False)
        )

        chart_col1, chart_col2 = st.columns([1.4, 1])

        with chart_col1:

            st.bar_chart(
                category_data,
                height=300
            )

        with chart_col2:

            pie_data = pd.DataFrame(
                {
                    "Category": category_data.index,
                    "Amount": category_data.values
                }
            )

            st.vega_lite_chart(
                pie_data,
                {
                    "mark": {
                        "type": "arc",
                        "innerRadius": 65
                    },
                    "encoding": {
                        "theta": {
                            "field": "Amount",
                            "type": "quantitative"
                        },
                        "color": {
                            "field": "Category",
                            "type": "nominal"
                        },
                        "tooltip": [
                            {
                                "field": "Category",
                                "type": "nominal"
                            },
                            {
                                "field": "Amount",
                                "type": "quantitative",
                                "format": ",.2f"
                            }
                        ]
                    },
                    "height": 280
                }
            )

    else:

        st.info("No category data available.")


    # -----------------------------
    # Recent Transactions
    # -----------------------------

    st.markdown(
        '<div class="section-title">Recent Transactions</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Your latest expenses</div>',
        unsafe_allow_html=True
    )

    if not df.empty:

        recent = df.sort_values(
            "Date",
            ascending=False
        ).head(5)

        icon_map = {
            "Food": "🍔",
            "Shopping": "🛍️",
            "Transport": "🚗",
            "Bills": "💡",
            "Entertainment": "🎬",
            "Health": "💊",
            "Education": "📚",
            "Other": "💰"
        }

        for _, row in recent.iterrows():

            category = str(
                row.get("Category", "Other")
            )

            icon = icon_map.get(
                category,
                "💰"
            )

            description = row.get(
                "Description",
                ""
            )

            if pd.isna(description) or str(description).strip() == "":
                description = category

            transaction_date = row["Date"]

            if pd.notna(transaction_date):

                formatted_date = pd.to_datetime(
                    transaction_date
                ).strftime("%Y-%m-%d")

            else:

                formatted_date = "Unknown date"

            amount = float(
                row["Amount"]
            )

            # Use columns instead of raw HTML for amount
            t1, t2, t3 = st.columns(
                [0.8, 5, 2]
            )

            with t1:

                st.markdown(
                    f"### {icon}"
                )

            with t2:

                st.markdown(
                    f"**{description}**"
                )

                st.caption(
                    f"{category} • {formatted_date}"
                )

            with t3:

                st.markdown(
                    f"**- Rs. {amount:,.2f}**"
                )


    else:

        st.info("No transactions available.")


    # -----------------------------
    # Budget + AI
    # -----------------------------

    st.markdown(
        '<div class="section-title">Financial Summary</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Budget and AI prediction</div>',
        unsafe_allow_html=True
    )

    budget_col, ai_col = st.columns(2)


    # Budget
    with budget_col:

        monthly_budget = 50000.0

        budget_spending = total_spending

        budget_remaining = max(
            monthly_budget - budget_spending,
            0
        )

        budget_used = (
            budget_spending / monthly_budget
        ) if monthly_budget > 0 else 0

        budget_used = min(
            max(budget_used, 0),
            1
        )

        st.markdown(
            """
            <div class="budget-card">
                <div class="budget-title">
                    Monthly Budget
                </div>
                <div class="budget-description">
                    Your current monthly budget
                </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="budget-value">
                Rs. {budget_remaining:,.0f}
            </div>
            <div class="budget-description">
                remaining from Rs. {monthly_budget:,.0f}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            budget_used
        )

        st.caption(
            f"{budget_used * 100:.1f}% of budget used"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # AI
    with ai_col:

        prediction = None

        if len(df) >= 2:

            try:

                prediction = predict_next_expense(
                    df
                )

            except Exception:

                prediction = None


        if prediction is not None:

            st.markdown(
                f"""
                <div class="ai-card">
                    <div class="ai-label">
                        🤖 AI PREDICTION
                    </div>
                    <div class="ai-value">
                        Rs. {prediction:,.2f}
                    </div>
                    <div class="ai-description">
                        Estimated next expense
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="ai-card">
                    <div class="ai-label">
                        🤖 AI PREDICTION
                    </div>
                    <div class="ai-value">
                        —
                    </div>
                    <div class="ai-description">
                        Add more expenses to generate a prediction
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# ADD EXPENSE
# =========================================================

elif page == "➕ Add Expense":

    st.markdown(
        '<div class="page-title">Add Expense</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">Record a new personal expense</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="content-card">',
        unsafe_allow_html=True
    )

    expense_date = st.date_input(
        "Date",
        value=date.today()
    )

    category = st.selectbox(
        "Category",
        [
            "Food",
            "Shopping",
            "Transport",
            "Bills",
            "Entertainment",
            "Health",
            "Education",
            "Other"
        ]
    )

    amount = st.number_input(
        "Amount (Rs.)",
        min_value=0.0,
        step=50.0,
        format="%.2f"
    )

    description = st.text_input(
        "Description",
        placeholder="Example: Lunch, Bus ticket, Groceries..."
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    if st.button(
        "➕ Add Expense",
        use_container_width=True
    ):

        if amount <= 0:

            st.error(
                "Please enter an amount greater than 0."
            )

        else:

            new_expense = pd.DataFrame(
                [
                    {
                        "Date": expense_date.strftime("%Y-%m-%d"),
                        "Category": category,
                        "Amount": amount,
                        "Description": description
                    }
                ]
            )

            new_expense.to_csv(
                FILE_NAME,
                mode="a",
                header=False,
                index=False
            )

            st.success(
                "Expense added successfully! 🎉"
            )

            st.rerun()


# =========================================================
# EXPENSE HISTORY
# =========================================================

elif page == "📜 Expense History":

    st.markdown(
        '<div class="page-title">Expense History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">View all your recorded expenses</div>',
        unsafe_allow_html=True
    )

    if not df.empty:

        display_df = df.copy()

        display_df["Date"] = display_df[
            "Date"
        ].dt.strftime(
            "%Y-%m-%d"
        )

        display_df["Amount"] = display_df[
            "Amount"
        ].apply(
            lambda x: f"Rs. {x:,.2f}"
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Summary")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Total Spending",
                f"Rs. {df['Amount'].sum():,.2f}"
            )

        with c2:
            st.metric(
                "Transactions",
                len(df)
            )

        with c3:
            st.metric(
                "Average",
                f"Rs. {df['Amount'].mean():,.2f}"
            )

        csv_data = df.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download Expense CSV",
            csv_data,
            file_name="expenses.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:

        st.info(
            "No expenses recorded yet."
        )


# =========================================================
# MONTHLY ANALYSIS
# =========================================================

elif page == "📊 Monthly Analysis":

    st.markdown(
        '<div class="page-title">Monthly Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">Understand your monthly spending patterns</div>',
        unsafe_allow_html=True
    )

    if not df.empty:

        monthly_df = df.copy()

        monthly_df["Date"] = pd.to_datetime(
            monthly_df["Date"],
            errors="coerce"
        )

        monthly_df = monthly_df.dropna(
            subset=["Date"]
        )

        monthly_df["Month"] = (
            monthly_df["Date"]
            .dt.to_period("M")
            .astype(str)
        )

        monthly_totals = (
            monthly_df
            .groupby("Month")["Amount"]
            .sum()
            .sort_index()
        )

        st.markdown(
            '<div class="section-title">Monthly Spending</div>',
            unsafe_allow_html=True
        )

        st.line_chart(
            monthly_totals,
            height=350
        )

        st.markdown(
            '<div class="section-title">Month-by-Month Comparison</div>',
            unsafe_allow_html=True
        )

        comparison_df = pd.DataFrame(
            {
                "Month": monthly_totals.index,
                "Total Spending": monthly_totals.values
            }
        )

        comparison_df["Total Spending"] = comparison_df[
            "Total Spending"
        ].apply(
            lambda x: f"Rs. {x:,.2f}"
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            '<div class="section-title">Category Distribution</div>',
            unsafe_allow_html=True
        )

        category_month = (
            df.groupby("Category")["Amount"]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(
            category_month,
            height=320
        )

    else:

        st.info(
            "Add expenses to view monthly analysis."
        )


# =========================================================
# AI INSIGHTS
# =========================================================

elif page == "🤖 AI Insights":

    st.markdown(
        '<div class="page-title">AI Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">Smart insights from your spending data</div>',
        unsafe_allow_html=True
    )

    if not df.empty:

        total = df["Amount"].sum()

        category_spending = (
            df.groupby("Category")["Amount"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        highest_category = (
            category_spending.index[0]
        )

        highest_category_amount = (
            category_spending.iloc[0]
        )

        category_percentage = (
            highest_category_amount / total * 100
            if total > 0
            else 0
        )

        st.markdown(
            """
            <div class="content-card">
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "### 🔍 Spending Insight"
        )

        st.write(
            f"Your highest spending category is **{highest_category}**."
        )

        st.write(
            f"You spent **Rs. {highest_category_amount:,.2f}** "
            f"on {highest_category}, which is "
            f"**{category_percentage:.1f}%** of your total spending."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">Category Spending</div>',
            unsafe_allow_html=True
        )

        st.bar_chart(
            category_spending,
            height=320
        )


        # Prediction

        st.markdown(
            '<div class="section-title">Future Expense Prediction</div>',
            unsafe_allow_html=True
        )

        if len(df) >= 2:

            try:

                prediction = predict_next_expense(
                    df
                )

                st.markdown(
                    f"""
                    <div class="ai-card">
                        <div class="ai-label">
                            🤖 AI PREDICTION
                        </div>
                        <div class="ai-value">
                            Rs. {prediction:,.2f}
                        </div>
                        <div class="ai-description">
                            Estimated next expense based on your recorded transactions
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:

                st.warning(
                    f"Prediction unavailable: {e}"
                )

        else:

            st.info(
                "Add at least 2 expenses to generate an AI prediction."
            )


        # Recommendation

        st.markdown(
            '<div class="section-title">💡 Smart Recommendation</div>',
            unsafe_allow_html=True
        )

        if category_percentage >= 50:

            st.warning(
                f"{highest_category} uses {category_percentage:.1f}% "
                "of your total spending. Consider reviewing this category."
            )

        elif category_percentage >= 30:

            st.info(
                f"{highest_category} represents {category_percentage:.1f}% "
                "of your spending. Keep monitoring this category."
            )

        else:

            st.success(
                "Your spending is distributed across multiple categories."
            )

    else:

        st.info(
            "Add some expenses to generate AI insights."
        )


# =========================================================
# BUDGET MANAGEMENT
# =========================================================

elif page == "💰 Budget Management":

    st.markdown(
        '<div class="page-title">Budget Management</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">Set and monitor your monthly spending budget</div>',
        unsafe_allow_html=True
    )

    monthly_budget = st.number_input(
        "Monthly Budget (Rs.)",
        min_value=1000.0,
        value=50000.0,
        step=1000.0,
        format="%.2f"
    )

    total_spending = (
        df["Amount"].sum()
        if not df.empty
        else 0
    )

    remaining = (
        monthly_budget - total_spending
    )

    percentage_used = (
        total_spending / monthly_budget * 100
        if monthly_budget > 0
        else 0
    )

    progress_value = (
        total_spending / monthly_budget
        if monthly_budget > 0
        else 0
    )

    progress_value = min(
        max(progress_value, 0),
        1
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Monthly Budget",
            f"Rs. {monthly_budget:,.0f}"
        )

    with col2:

        st.metric(
            "Spent",
            f"Rs. {total_spending:,.0f}"
        )

    with col3:

        st.metric(
            "Remaining",
            f"Rs. {max(remaining, 0):,.0f}"
        )

    st.markdown(
        '<div class="section-title">Budget Usage</div>',
        unsafe_allow_html=True
    )

    st.progress(
        progress_value
    )

    st.write(
        f"**{percentage_used:.1f}%** of your monthly budget used"
    )

    if percentage_used < 50:

        st.success(
            "You are comfortably within your budget. 👍"
        )

    elif percentage_used < 80:

        st.info(
            "Your spending is getting closer to your budget limit."
        )

    elif percentage_used < 100:

        st.warning(
            "You are approaching your monthly budget limit."
        )

    else:

        st.error(
            "Your spending has reached or exceeded your monthly budget."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        FinTrack AI • Smart Personal Finance Dashboard
    </div>
    """,
    unsafe_allow_html=True
)