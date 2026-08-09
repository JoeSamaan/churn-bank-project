



import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.metrics import confusion_matrix, classification_report
from Bank import results
from tensorflow.keras.models import load_model

model = load_model("DeepLearning.keras")
from Bank import (
    comp,
    dataclean,
    model,
    X_column,
    scaler,
    history,
    PredLG,
    PredDT,
    PredRF,
    PredXGB,
    PredDL,
    LG,
    DT,
    RF,
    XGB,
    reportLG,
    reportDT,
    reportDL,
    reportRF,
    reportXGB,
    df,
    Importance,
    y_test
    )


st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# ===========================
# Sidebar
# ===========================

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Data Analysis",
        "Model Comparison",
        "Deep Learning (TensorFlow)",
        "Prediction",
        "About Me"
    ]
)

# ===========================
# Home
# ===========================

if page == "Home":
    st.markdown(""" لاحظ ان الشغل في البنوك حرام 😂😘
    """)
    st.title("🏦 Bank Customer Churn Prediction")

    st.markdown("---")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Models",5)
    c2.metric("Features",len(X_column))
    c3.metric("Dataset","10000")
    c4.metric("Target","Exited")

    st.markdown("---")

    st.subheader("Models Used")

    st.write("""
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - XGBoost
    - Deep Learning (TensorFlow)
    """)




# ===========================
# Data Analysis
# ===========================


if page == "Data Analysis":
    st.subheader("📈 Feature Relationships")

    col5, col6 = st.columns(2)

    with col5:

        st.markdown("#### Age vs Balance")

        fig5, ax5 = plt.subplots(figsize=(5,4))

        sns.scatterplot(
            data=dataclean,
            x="Age",
            y="Balance",
            hue="Exited",
            alpha=0.6,
            ax=ax5
        )

        plt.tight_layout()

        st.pyplot(fig5)


    with col6:

        st.markdown("#### Credit Score vs Balance")

        fig6, ax6 = plt.subplots(figsize=(5,4))

        sns.scatterplot(
            data=dataclean,
            x="CreditScore",
            y="Balance",
            hue="Exited",
            alpha=0.6,
            ax=ax6
        )

        plt.tight_layout()

        st.pyplot(fig6)


    col7, col8 = st.columns(2)

    with col7:

        st.markdown("#### Age vs Exited")

        fig7, ax7 = plt.subplots(figsize=(5,4))

        sns.boxplot(
            data=dataclean,
            x="Exited",
            y="Age",
            ax=ax7
        )

        st.pyplot(fig7)


    with col8:

        st.markdown("#### Balance vs Exited")

        fig8, ax8 = plt.subplots(figsize=(5,4))

        sns.boxplot(
            data=dataclean,
            x="Exited",
            y="Balance",
            ax=ax8
        )

        st.pyplot(fig8)
    st.subheader("🔥 Correlation Heatmap")

    fig9, ax9 = plt.subplots(figsize=(10,6))

    sns.heatmap(
        dataclean.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax9
    )

    plt.tight_layout()

    st.pyplot(fig9)




    st.subheader("📊 Statistical Summary")
    st.write(dataclean.describe())

    col14, col15 ,col16, col17 = st.columns(4)

    with col14:
        st.subheader("Bank Churn Distribution")
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        dataclean["Exited"].value_counts().plot(kind="bar", ax=ax1)
        plt.tight_layout()
        st.pyplot(fig1)

    with col15:
        st.subheader("📉 Age Distribution")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.hist(dataclean["Age"], bins=30)
        plt.tight_layout()
        st.pyplot(fig2)



    # col3, col4 = st.columns(2)

    with col16:
        st.subheader("Balance Distribution")
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        ax3.hist(dataclean["Balance"], bins=30)
        plt.tight_layout()
        st.pyplot(fig3)

    with col17:
        st.subheader("Card Type vs Exited ")
        #st.subheader("")
        
        fig4, ax4 = plt.subplots(figsize=(5, 4))

        sns.countplot(
            data=df,
            x="Card Type",
            hue="Exited",
            ax=ax4
        )

        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig4)

    col22, = st.columns(1)

    with col22:
            st.subheader("Feature Importance")

            fig22, ax22 = plt.subplots(figsize=(10, 8))

            plot_importance(
                    XGB,
                    ax=ax22,
                    importance_type="gain",
                    show_values=True,
                    grid=False
                )

            for text in ax22.texts:
                text.set_text(f"{float(text.get_text()):.2f}")
                text.set_fontsize(10)


    st.pyplot(fig22)





# ===========================
# Comparison
# ===========================

# elif page == "Model Comparison":

#     st.title("📊 Model Comparison")

#     st.dataframe(comp,use_container_width=True)

#     metric = st.selectbox(
#         "Choose Metric",
#         [
#             "accuracy",
#             "precision",
#             "recall",
#             "f1"
#         ]
#     )

#     fig,ax = plt.subplots(figsize=(8,5))

#     ax.bar(
#         comp["Model"],
#         comp[metric]
#     )

#     ax.set_ylabel(metric)

#     st.pyplot(fig)





elif page == "Model Comparison":

    st.title("📊 Model Performance Comparison")

    st.dataframe(comp, use_container_width=True)

    # ===========================
    # Metrics Comparison
    # ===========================

    st.subheader("📈 Model Metrics")

    metric = st.selectbox(
        "Choose Metric",
        ["accuracy", "precision", "recall", "f1"]
    )

    fig, ax = plt.subplots(figsize=(3.8, 2.5))

    ax.bar(comp["Model"], comp[metric], color="skyblue")

    ax.set_ylim(0, 1)

    ax.set_ylabel(metric.capitalize(), fontsize=8)

    ax.set_title(metric.capitalize(), fontsize=9)

    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)

    plt.xticks(rotation=20)

    plt.tight_layout()

    # st.pyplot(fig)
    st.pyplot(fig, use_container_width=False)

    # ===========================
    # Accuracy & Recall
    # ===========================

    st.subheader("Accuracy vs Recall")

    col1, col2 = st.columns(2)

    with col1:

        fig1, ax1 = plt.subplots(figsize=(4,3))

        ax1.bar(comp["Model"], comp["accuracy"], color="royalblue")

        ax1.set_ylim(0,1)

        ax1.set_title("Accuracy")
        ax.tick_params(axis='x', labelsize=5)
        ax.tick_params(axis='y', labelsize=5)
        plt.xticks(rotation=20)

        plt.tight_layout()

        st.pyplot(fig1)


    with col2:

        fig2, ax2 = plt.subplots(figsize=(4,3))

        ax2.bar(comp["Model"], comp["recall"], color="tomato")

        ax2.set_ylim(0,1)

        ax2.set_title("Recall")

        plt.xticks(rotation=15)

        plt.tight_layout()

        st.pyplot(fig2)

    # ===========================
    # Confusion Matrix
    # ===========================

    st.subheader("📋 Confusion Matrices")

    models = {
        "Logistic Regression": PredLG,
        "Decision Tree": PredDT,
        "Random Forest": PredRF,
        "XGBoost": PredXGB,
        "Deep Learning (TensorFlow)": PredDL
    }

    cols = st.columns(2)

    for i, (name, pred) in enumerate(models.items()):

        with cols[i % 2]:

            st.markdown(f"### {name}")

            cm = confusion_matrix(y_test, pred)

            fig3, ax3 = plt.subplots(figsize=(3,2))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                ax=ax3
            )

            ax3.set_xlabel("Predicted")

            ax3.set_ylabel("Actual")

            st.pyplot(fig3)

            report = classification_report(
                y_test,
                pred,
                output_dict=True
            )

            report = pd.DataFrame(report).transpose()

            st.dataframe(
                report.style
                .background_gradient(cmap="Blues")
                .format("{:.3f}"),
                use_container_width=True
            )












# # ===========================
# # Deep Learning
# # ===========================

# elif page=="Deep Learning":

#     st.title("🧠 Deep Learning")

#     fig,ax = plt.subplots(figsize=(5,3))

#     ax.plot(history.history["accuracy"],label="Train")

#     ax.plot(history.history["val_accuracy"],label="Validation")

#     ax.legend()

#     ax.set_title("Accuracy")

#     st.pyplot(fig)

#     fig2,ax2 = plt.subplots(figsize=(5,3))

#     ax2.plot(history.history["loss"],label="Train")

#     ax2.plot(history.history["val_loss"],label="Validation")

#     ax2.legend()

#     ax2.set_title("Loss")

#     st.pyplot(fig2)




elif page == "Deep Learning (TensorFlow)":

    st.title("🧠 Deep Learning (TensorFlow) Performance")

    st.markdown("---")

    # ================= Metrics =================

    c1, c2 = st.columns(2)

    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]

    train_loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]

    c1.metric("Training Accuracy", f"{train_acc:.2%}")
    c2.metric("Validation Accuracy", f"{val_acc:.2%}")

    st.markdown("---")

    # ================= Charts =================

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(figsize=(3,2))

        ax.plot(history.history["accuracy"], linewidth=2, label="Train")
        ax.plot(history.history["val_accuracy"], linewidth=2, label="Validation")

        ax.set_title("Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.3)
        ax.legend()

        st.pyplot(fig, use_container_width=False)

    with col2:

        fig2, ax2 = plt.subplots(figsize=(3,2))

        ax2.plot(history.history["loss"], linewidth=2, label="Train")
        ax2.plot(history.history["val_loss"], linewidth=2, label="Validation")

        ax2.set_title("Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.grid(alpha=0.3)
        ax2.legend()

        st.pyplot(fig2, use_container_width=False)

    st.markdown("---")

    # ================= Training Summary =================

    st.subheader("📋 Training Summary")

    summary = pd.DataFrame({
        "Metric": [
            "Training Accuracy",
            "Validation Accuracy",
            "Training Loss",
            "Validation Loss",
            "Epochs"
        ],
        "Value": [
            f"{train_acc:.4f}",
            f"{val_acc:.4f}",
            f"{train_loss:.4f}",
            f"{val_loss:.4f}",
            len(history.history["accuracy"])
        ]
    })

    st.dataframe(summary, use_container_width=True)





    st.subheader("📈 Learning Curve")

    fig3, ax3 = plt.subplots(figsize=(4,2))

    ax3.plot(history.history["accuracy"], label="Train Accuracy")
    ax3.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax3.plot(history.history["loss"], label="Train Loss")
    ax3.plot(history.history["val_loss"], label="Validation Loss")

    ax3.legend()
    ax3.grid(alpha=0.3)

    # st.pyplot(fig3)
    st.pyplot(fig3, use_container_width=False)
























# ===========================
# Prediction
# ===========================

elif page=="Prediction":

    st.title("🔮 Customer Churn Prediction")

    model_name = st.selectbox(
        "Choose Model",
        list(model.keys())
    )

    CreditScore = st.number_input("Credit Score",300,900,650)

    Geography = st.selectbox(
        "Geography",
        ["Germany","France","Spain"]
    )
    # if Geography == "Germany":
    #     Geography = 0
    # elif Geography == "France":
    #     Geography = 1
    # else:
    #     Geography= 2
        
        
    Gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )
    if Gender == "Male":
        Gender = 1
    else:
        Gender = 0



    Age = st.slider(
        "Age",
        18,
        100,
        30
    )

    Tenure = st.slider(
        "Tenure",
        0,
        10,
        5
    )

    Balance = st.number_input(
        "Balance",
        0.0,
        300000.0,
        50000.0
    )

    NumOfProducts = st.slider(
        "Products",
        1,
        4,
        1
    )

    HasCrCard = st.selectbox(
        "Has Credit Card",
        ["No","Yes"]
    )
    if HasCrCard == "Yes":
        HasCrCard = 1
    else:
        HasCrCard = 0



    IsActiveMember = st.selectbox(
        "Active Member",
        ["No","Yes"]
    )

    if IsActiveMember == "Yes":
        IsActiveMember = 1
    else:
        IsActiveMember = 0


    EstimatedSalary = st.number_input(
        "Estimated Salary",
        0.0,
        300000.0,
        50000.0
    )

    Satisfaction = st.slider(
        "Satisfaction",
        1,
        5,
        3
    )

    CardType = st.selectbox(
        "Card Type",
        ["DIAMOND","GOLD","SILVER","PLATINUM"]
    )


    
    # if CardType == "DIAMOND":
    #     CardType = 0
    # elif CardType == "GOLD":
    #     CardType = 1
    # elif CardType == "SILVER":
    #     CardType = 2
    # else:
    #     CardType= 3
        



    






    PointEarned = st.number_input(
        "Point Earned",
        0,
        1000,
        400
    )

    if st.button("Predict"):

        data = pd.DataFrame(columns=X_column)

        data.loc[0] = 0

        data["CreditScore"] = CreditScore
        data["Gender"] = Gender
        data["Age"] = Age
        data["Tenure"] = Tenure
        data["Balance"] = Balance
        data["NumOfProducts"] = NumOfProducts
        data["HasCrCard"] = HasCrCard
        data["IsActiveMember"] = IsActiveMember
        data["EstimatedSalary"] = EstimatedSalary
        data["Satisfaction Score"] = Satisfaction
        data["Point Earned"] = PointEarned

        data["Geography_Germany"] = Geography == "Germany"
        data["Geography_Spain"] = Geography == "Spain"

        data["Card Type_GOLD"] = CardType == "GOLD"
        data["Card Type_PLATINUM"] = CardType == "PLATINUM"
        data["Card Type_SILVER"] = CardType == "SILVER"

        if model_name in [
            "Logistic Regression",
            "Deep Learning"
        ]:

            data_scaled = scaler.transform(data)

        if model_name=="Deep Learning":

            prob = model[model_name].predict(data_scaled)[0][0]

            pred = 1 if prob>0.5 else 0

        else:

            if model_name=="Logistic Regression":

                pred = model[model_name].predict(data_scaled)[0]

                prob = model[model_name].predict_proba(data_scaled)[0][1]

            else:

                pred = model[model_name].predict(data)[0]

                prob = model[model_name].predict_proba(data)[0][1]

        st.markdown("---")

        if pred==1:

            st.error(f"⚠ Customer Will Churn\n\nProbability = {prob:.2%}")

        else:

            st.success(f"✅ Customer Will Stay\n\nProbability = {(1-prob):.2%}")



elif page == "About Me":
    from PIL import Image

    st.title("About Me")

    st.markdown("---")

    col1, col2 = st.columns([1,2])

    with col1:

        image = Image.open("abcdefghj.png")

        st.image(image, width=560)

    with col2:

        st.markdown("## Developer")

        st.markdown("### **Joseph Samaan**")

        st.markdown("""
**🎓 Faculty of Commerce**  
Public Policy Information Systems

---

### 🤖 Machine Learning Project

This project predicts whether a customer is likely to leave the bank using several Machine Learning and Deep Learning (TensorFlow)  algorithms.

---

### 🧠 Models Used

- ✅ Logistic Regression
- ✅ Decision Tree
- ✅ Random Forest
- ✅ XGBoost
- ✅ Deep Learning TensorFlow / Keras (ANN)

---

### 🛠️ Libraries

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Matplotlib
- Seaborn
- Streamlit
""")

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Models", "5")
    c2.metric("Dataset", "10,000")
    c3.metric("Features", "13")
    c4.metric("Target", "Exited")



st.markdown(
    """
    
    <div style="text-align:center; color:gray; font-size:16px;">
        ❤️ Developed by <b>Joseph Samaan</b><br>
        © 2026 All Rights Reserved
    </div>
    """,
    unsafe_allow_html=True
)




























# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report

# from Bank import (
#     comp, dataclean, model, X_column, scaler, history,
#     PredLG, PredDT, PredRF, PredXGB, PredDL,
#     LG, DT, RF, XGB,
#     reportLG, reportDT, reportDL, reportRF, reportXGB,
#     y_test,
# )

# # =========================================================
# # Page Config
# # =========================================================
# st.set_page_config(
#     page_title="Bank Churn Prediction",
#     page_icon="🏦",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # =========================================================
# # Custom CSS
# # =========================================================
# st.markdown(
#     """
#     <style>
#         /* Main background */
#         .stApp { background: #f8fafc; }

#         /* Sidebar */
#         section[data-testid="stSidebar"] {
#             background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
#         }
#         section[data-testid="stSidebar"] .stMarkdown,
#         section[data-testid="stSidebar"] label {
#             color: #e2e8f0 !important;
#         }
#         section[data-testid="stSidebar"] .stRadio label {
#             color: #cbd5e1 !important;
#             font-size: 14px !important;
#             font-weight: 500 !important;
#         }
#         section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]:checked + div {
#             color: #60a5fa !important;
#         }

#         /* Titles */
#         h1 { color: #0f172a !important; font-weight: 700 !important; }
#         h2 { color: #1e293b !important; font-weight: 600 !important; }
#         h3 { color: #334155 !important; font-weight: 600 !important; }

#         /* Cards (metrics) */
#         div[data-testid="stMetric"] {
#             background: #ffffff;
#             border: 1px solid #e2e8f0;
#             border-radius: 12px;
#             padding: 16px 20px;
#             box-shadow: 0 1px 3px rgba(0,0,0,0.04);
#         }
#         div[data-testid="stMetric"] label {
#             color: #64748b !important;
#             font-size: 12px !important;
#             font-weight: 600 !important;
#             text-transform: uppercase;
#             letter-spacing: 0.5px;
#         }
#         div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
#             color: #0f172a !important;
#             font-size: 28px !important;
#             font-weight: 700 !important;
#         }

#         /* Buttons */
#         .stButton > button {
#             background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
#             color: white !important;
#             border: none !important;
#             border-radius: 10px !important;
#             padding: 10px 24px !important;
#             font-weight: 600 !important;
#             box-shadow: 0 4px 12px rgba(37,99,235,0.25) !important;
#             transition: all 0.2s ease !important;
#         }
#         .stButton > button:hover {
#             transform: translateY(-1px);
#             box-shadow: 0 6px 16px rgba(37,99,235,0.35) !important;
#         }

#         /* Inputs */
#         .stTextInput input, .stNumberInput input, .stSelectbox select {
#             border-radius: 8px !important;
#             border: 1px solid #e2e8f0 !important;
#         }
#         .stTextInput input:focus, .stNumberInput input:focus {
#             border-color: #3b82f6 !important;
#             box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
#         }

#         /* Footer */
#         .footer {
#             text-align: center;
#             color: #94a3b8;
#             font-size: 13px;
#             padding: 20px 0;
#             border-top: 1px solid #e2e8f0;
#             margin-top: 40px;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # =========================================================
# # Sidebar Navigation
# # =========================================================
# with st.sidebar:
#     st.markdown("### 🏦 BankChurn")
#     st.caption("ML Prediction Dashboard")
#     st.markdown("---")
#     page = st.radio(
#         "Navigation",
#         ["Home", "Data Analysis", "Model Comparison", "Deep Learning", "Prediction", "About Me"],
#         label_visibility="collapsed",
#     )
#     st.markdown("---")
#     st.caption("📊 5 Models • 13 Features • 10K Samples")

# # =========================================================
# # HOME
# # =========================================================
# if page == "Home":
#     st.markdown(
#         """
#         <div style="background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
#                     padding: 40px; border-radius: 16px; margin-bottom: 24px;">
#             <h1 style="color: white; margin: 0; font-size: 32px;">🏦 Bank Customer Churn Prediction</h1>
#             <p style="color: #93c5fd; margin-top: 8px; font-size: 15px;">
#                 Predict whether a customer is likely to leave the bank using ML &amp; Deep Learning
#             </p>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Models", "5")
#     c2.metric("Features", len(X_column))
#     c3.metric("Dataset", "10,000")
#     c4.metric("Target", "Exited")

#     st.markdown("---")

#     col_left, col_right = st.columns([2, 1])
#     with col_left:
#         st.subheader("🤖 Models Used")
#         st.markdown(
#             """
#             - ✅ **Logistic Regression** — Linear baseline
#             - ✅ **Decision Tree** — Interpretable tree
#             - ✅ **Random Forest** — Ensemble of trees
#             - ✅ **XGBoost** — Gradient boosted (best)
#             - ✅ **Deep Learning** — Artificial Neural Network
#             """
#         )
#     with col_right:
#         st.subheader("🎯 Best Model")
#         best = comp.loc[comp["accuracy"].idxmax()]
#         st.metric("Top Accuracy", f"{best['accuracy']:.2%}", best["Model"])

# # =========================================================
# # DATA ANALYSIS
# # =========================================================
# elif page == "Data Analysis":
#     st.title("📈 Data Analysis")
#     st.markdown("---")

#     st.subheader("Feature Relationships")

#     col5, col6 = st.columns(2)
#     with col5:
#         st.markdown("#### Age vs Balance")
#         fig5, ax5 = plt.subplots(figsize=(6, 4))
#         sns.scatterplot(data=dataclean, x="Age", y="Balance", hue="Exited",
#                         alpha=0.6, palette={0: "#3b82f6", 1: "#ef4444"}, ax=ax5)
#         plt.tight_layout()
#         st.pyplot(fig5, use_container_width=True)

#     with col6:
#         st.markdown("#### Credit Score vs Balance")
#         fig6, ax6 = plt.subplots(figsize=(6, 4))
#         sns.scatterplot(data=dataclean, x="CreditScore", y="Balance", hue="Exited",
#                         alpha=0.6, palette={0: "#3b82f6", 1: "#ef4444"}, ax=ax6)
#         plt.tight_layout()
#         st.pyplot(fig6, use_container_width=True)

#     col7, col8 = st.columns(2)
#     with col7:
#         st.markdown("#### Age vs Exited")
#         fig7, ax7 = plt.subplots(figsize=(6, 4))
#         sns.boxplot(data=dataclean, x="Exited", y="Age", palette="Set2", ax=ax7)
#         st.pyplot(fig7, use_container_width=True)

#     with col8:
#         st.markdown("#### Balance vs Exited")
#         fig8, ax8 = plt.subplots(figsize=(6, 4))
#         sns.boxplot(data=dataclean, x="Exited", y="Balance", palette="Set2", ax=ax8)
#         st.pyplot(fig8, use_container_width=True)

#     st.subheader("🔥 Correlation Heatmap")
#     fig9, ax9 = plt.subplots(figsize=(12, 8))
#     sns.heatmap(dataclean.corr(), annot=True, cmap="coolwarm", fmt=".2f",
#                 linewidths=0.5, ax=ax9)
#     plt.tight_layout()
#     st.pyplot(fig9, use_container_width=True)

#     st.subheader("📊 Statistical Summary")
#     st.dataframe(dataclean.describe(), use_container_width=True)

#     col14, col15, col16, col17 = st.columns(4)
#     with col14:
#         st.markdown("#### Churn Distribution")
#         fig1, ax1 = plt.subplots(figsize=(5, 3.5))
#         dataclean["Exited"].value_counts().plot(kind="bar", color=["#3b82f6", "#ef4444"], ax=ax1)
#         plt.tight_layout()
#         st.pyplot(fig1, use_container_width=True)

#     with col15:
#         st.markdown("#### Age Distribution")
#         fig2, ax2 = plt.subplots(figsize=(5, 3.5))
#         ax2.hist(dataclean["Age"], bins=30, color="#6366f1", edgecolor="white")
#         plt.tight_layout()
#         st.pyplot(fig2, use_container_width=True)

#     with col16:
#         st.markdown("#### Balance Distribution")
#         fig3, ax3 = plt.subplots(figsize=(5, 3.5))
#         ax3.hist(dataclean["Balance"], bins=30, color="#06b6d4", edgecolor="white")
#         plt.tight_layout()
#         st.pyplot(fig3, use_container_width=True)

#     with col17:
#         st.markdown("#### Card Type vs Exited")
#         fig4, ax4 = plt.subplots(figsize=(6, 4))
#         sns.countplot(data=dataclean, x="Card Type", hue="Exited",
#                       palette={0: "#3b82f6", 1: "#ef4444"}, ax=ax4)
#         plt.xticks(rotation=15)
#         plt.tight_layout()
#         st.pyplot(fig4, use_container_width=True)

# # =========================================================
# # MODEL COMPARISON
# # =========================================================
# elif page == "Model Comparison":
#     st.title("📊 Model Performance Comparison")
#     st.markdown("---")

#     st.dataframe(comp, use_container_width=True)

#     st.subheader("📈 Model Metrics")
#     metric = st.selectbox("Choose Metric", ["accuracy", "precision", "recall", "f1"])

#     fig, ax = plt.subplots(figsize=(8, 4))
#     bars = ax.bar(comp["Model"], comp[metric], color="#3b82f6", edgecolor="white")
#     ax.set_ylim(0, 1)
#     ax.set_ylabel(metric.capitalize(), fontsize=11)
#     ax.set_title(metric.capitalize(), fontsize=13, fontweight="bold")
#     plt.xticks(rotation=20, ha="right")
#     for bar in bars:
#         ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
#                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
#     plt.tight_layout()
#     st.pyplot(fig, use_container_width=True)

#     st.subheader("Accuracy vs Recall")
#     col1, col2 = st.columns(2)
#     with col1:
#         fig1, ax1 = plt.subplots(figsize=(6, 4))
#         ax1.bar(comp["Model"], comp["accuracy"], color="#2563eb", edgecolor="white")
#         ax1.set_ylim(0, 1)
#         ax1.set_title("Accuracy", fontweight="bold")
#         plt.xticks(rotation=20, ha="right")
#         plt.tight_layout()
#         st.pyplot(fig1, use_container_width=True)

#     with col2:
#         fig2, ax2 = plt.subplots(figsize=(6, 4))
#         ax2.bar(comp["Model"], comp["recall"], color="#ef4444", edgecolor="white")
#         ax2.set_ylim(0, 1)
#         ax2.set_title("Recall", fontweight="bold")
#         plt.xticks(rotation=20, ha="right")
#         plt.tight_layout()
#         st.pyplot(fig2, use_container_width=True)

#     st.subheader("📋 Confusion Matrices")
#     models = {
#         "Logistic Regression": PredLG,
#         "Decision Tree": PredDT,
#         "Random Forest": PredRF,
#         "XGBoost": PredXGB,
#         "Deep Learning": PredDL,
#     }
#     cols = st.columns(2)
#     for i, (name, pred) in enumerate(models.items()):
#         with cols[i % 2]:
#             st.markdown(f"### {name}")
#             cm = confusion_matrix(y_test, pred)
#             fig3, ax3 = plt.subplots(figsize=(4, 3))
#             sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#                         cbar=False, ax=ax3,
#                         xticklabels=["Stay", "Churn"],
#                         yticklabels=["Stay", "Churn"])
#             ax3.set_xlabel("Predicted")
#             ax3.set_ylabel("Actual")
#             st.pyplot(fig3, use_container_width=True)

#             report = classification_report(y_test, pred, output_dict=True)
#             report = pd.DataFrame(report).transpose()
#             st.dataframe(
#                 report.style.background_gradient(cmap="Blues").format("{:.3f}"),
#                 use_container_width=True,
#             )

# # =========================================================
# # DEEP LEARNING
# # =========================================================
# elif page == "Deep Learning":
#     st.title("🧠 Deep Learning Performance")
#     st.markdown("---")

#     train_acc = history.history["accuracy"][-1]
#     val_acc = history.history["val_accuracy"][-1]
#     train_loss = history.history["loss"][-1]
#     val_loss = history.history["val_loss"][-1]

#     c1, c2 = st.columns(2)
#     c1.metric("Training Accuracy", f"{train_acc:.2%}")
#     c2.metric("Validation Accuracy", f"{val_acc:.2%}")

#     st.markdown("---")

#     col1, col2 = st.columns(2)
#     with col1:
#         fig, ax = plt.subplots(figsize=(6, 4))
#         ax.plot(history.history["accuracy"], linewidth=2, label="Train", color="#10b981")
#         ax.plot(history.history["val_accuracy"], linewidth=2, label="Validation", color="#3b82f6")
#         ax.set_title("Accuracy", fontweight="bold")
#         ax.set_xlabel("Epoch")
#         ax.set_ylabel("Accuracy")
#         ax.grid(alpha=0.3)
#         ax.legend()
#         st.pyplot(fig, use_container_width=True)

#     with col2:
#         fig2, ax2 = plt.subplots(figsize=(6, 4))
#         ax2.plot(history.history["loss"], linewidth=2, label="Train", color="#f59e0b")
#         ax2.plot(history.history["val_loss"], linewidth=2, label="Validation", color="#ef4444")
#         ax2.set_title("Loss", fontweight="bold")
#         ax2.set_xlabel("Epoch")
#         ax2.set_ylabel("Loss")
#         ax2.grid(alpha=0.3)
#         ax2.legend()
#         st.pyplot(fig2, use_container_width=True)

#     st.markdown("---")
#     st.subheader("📋 Training Summary")
#     summary = pd.DataFrame({
#         "Metric": ["Training Accuracy", "Validation Accuracy", "Training Loss", "Validation Loss", "Epochs"],
#         "Value": [f"{train_acc:.4f}", f"{val_acc:.4f}", f"{train_loss:.4f}", f"{val_loss:.4f}", len(history.history["accuracy"])],
#     })
#     st.dataframe(summary, use_container_width=True)

#     st.subheader("📈 Learning Curve")
#     fig3, ax3 = plt.subplots(figsize=(8, 4))
#     ax3.plot(history.history["accuracy"], label="Train Accuracy", color="#10b981")
#     ax3.plot(history.history["val_accuracy"], label="Val Accuracy", color="#3b82f6")
#     ax3.plot(history.history["loss"], label="Train Loss", color="#f59e0b")
#     ax3.plot(history.history["val_loss"], label="Val Loss", color="#ef4444")
#     ax3.legend()
#     ax3.grid(alpha=0.3)
#     st.pyplot(fig3, use_container_width=True)

# # =========================================================
# # PREDICTION
# # =========================================================
# elif page == "Prediction":
#     st.title("🔮 Customer Churn Prediction")
#     st.markdown("---")

#     model_name = st.selectbox("Choose Model", list(model.keys()))

#     col_a, col_b, col_c = st.columns(3)
#     with col_a:
#         CreditScore = st.number_input("Credit Score", 300, 900, 650)
#         Geography = st.selectbox("Geography", ["Germany", "France", "Spain"])
#         Geography = {"Germany": 0, "France": 1, "Spain": 2}[Geography]
#         Gender = st.selectbox("Gender", ["Male", "Female"])
#         Gender = {"Male": 1, "Female": 0}[Gender]

#     with col_b:
#         Age = st.slider("Age", 18, 100, 30)
#         Tenure = st.slider("Tenure", 0, 10, 5)
#         Balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
#         NumOfProducts = st.slider("Products", 1, 4, 1)

#     with col_c:
#         HasCrCard = st.selectbox("Has Credit Card", ["No", "Yes"])
#         HasCrCard = {"Yes": 1, "No": 0}[HasCrCard]
#         IsActiveMember = st.selectbox("Active Member", ["No", "Yes"])
#         IsActiveMember = {"Yes": 1, "No": 0}[IsActiveMember]
#         EstimatedSalary = st.number_input("Estimated Salary", 0.0, 300000.0, 50000.0)

#     col_d, col_e, col_f = st.columns(3)
#     with col_d:
#         Satisfaction = st.slider("Satisfaction", 1, 5, 3)
#     with col_e:
#         CardType = st.selectbox("Card Type", ["DIAMOND", "GOLD", "SILVER", "PLATINUM"])
#         CardType = {"DIAMOND": 0, "GOLD": 1, "SILVER": 2, "PLATINUM": 3}[CardType]
#     with col_f:
#         PointEarned = st.number_input("Point Earned", 0, 1000, 400)

#     st.markdown("---")

#     if st.button("🔮 Predict Churn", use_container_width=True):
#         data = pd.DataFrame([[
#             CreditScore, Geography, Gender, Age, Tenure, Balance,
#             NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary,
#             Satisfaction, CardType, PointEarned
#         ]], columns=X_column)

#         needs_scaling = model_name in ["Logistic Regression", "Deep Learning"]
#         if needs_scaling:
#             data_scaled = scaler.transform(data)

#         if model_name == "Deep Learning":
#             prob = model[model_name].predict(data_scaled)[0][0]
#             pred = 1 if prob > 0.5 else 0
#         elif model_name == "Logistic Regression":
#             pred = model[model_name].predict(data_scaled)[0]
#             prob = model[model_name].predict_proba(data_scaled)[0][1]
#         else:
#             pred = model[model_name].predict(data)[0]
#             prob = model[model_name].predict_proba(data)[0][1]

#         st.markdown("---")
#         if pred == 1:
#             st.error(f"⚠️ **Customer Will Churn**  \n\nProbability = {prob:.2%}")
#         else:
#             st.success(f"✅ **Customer Will Stay**  \n\nProbability = {(1 - prob):.2%}")

# # =========================================================
# # ABOUT ME
# # =========================================================
# elif page == "About Me":
#     st.title("About Me")
#     st.markdown("---")

#     col1, col2 = st.columns([1, 2])
#     with col1:
#         try:
#             from PIL import Image
#             image = Image.open("abcdefghj.png")
#             st.image(image, use_container_width=True)
#         except Exception:
#             st.markdown(
#                 """
#                 <div style="background: linear-gradient(135deg, #2563eb, #4f46e5);
#                             aspect-ratio: 1; border-radius: 16px;
#                             display: flex; align-items: center; justify-content: center;
#                             color: white; font-size: 64px; font-weight: bold;">
#                     JS
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )

#     with col2:
#         st.markdown("## Developer")
#         st.markdown("### **Joseph Samaan**")
#         st.markdown(
#             """
#             **🎓 Faculty of Commerce**
#             Public Policy Information Systems

#             ---

#             ### 🤖 Machine Learning Project
#             This project predicts whether a customer is likely to leave the bank using several
#             Machine Learning and Deep Learning algorithms.

#             ---

#             ### 🧠 Models Used
#             - ✅ Logistic Regression
#             - ✅ Decision Tree
#             - ✅ Random Forest
#             - ✅ XGBoost
#             - ✅ Deep Learning (ANN)

#             ---

#             ### 🛠️ Libraries
#             - Pandas · NumPy · Scikit-learn
#             - XGBoost · TensorFlow / Keras
#             - Matplotlib · Seaborn · Streamlit
#             """
#         )

#     st.markdown("---")
#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Models", "5")
#     c2.metric("Dataset", "10,000")
#     c3.metric("Features", "13")
#     c4.metric("Target", "Exited")

# # =========================================================
# # Footer
# # =========================================================
# st.markdown(
#     """
#     <div class="footer">
#         ❤️ Developed by <b>Joseph Samaan</b><br>
#         © 2026 All Rights Reserved
#     </div>
#     """,
#     unsafe_allow_html=True,
# )