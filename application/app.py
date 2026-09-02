import streamlit as st

from application.customer_segmentation import CustomerIntelligence
from application.customer_data import FEATURE_NAMES

st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="🤓",
    layout="wide"
)

st.title("🤓 Customer Intelligence & Segmentation System")
st.caption("From-scratch K-Means + KNN for customer segmentation and classification")

@st.cache_resource
def load_system():
    system = CustomerIntelligence(k=3)
    system.segment_customers()
    system.train_knn()
    return system

system = load_system()

result = system.segment_customers()

# ---------- Overview ----------
st.header("1. Customer Segmentation Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Customer Records", len(system.normalized_data))
c2.metric("Clusters", system.k)
c3.metric("Silhouette Score", f"{result['silhouette']:.3f}")

st.info(
    "The system uses the project's own K-Means implementation to discover "
    "customer groups without using sklearn K-Means."
)

# ---------- Cluster profiles ----------
st.subheader("Customer Segment Profiles")

profile_rows = []

for cluster, profile in system.cluster_profiles.items():
    averages = profile["averages"]
    profile_rows.append({
        "Cluster": f"Cluster {cluster}",
        "Customers": profile["size"],
        "Avg Age": round(averages[0], 1),
        "Avg Income": f"${averages[1]:,.0f}",
        "Avg Spending": round(averages[2], 1),
        "Avg Frequency": round(averages[3], 1)
    })

st.dataframe(profile_rows, use_container_width=True, hide_index=True)

# ---------- New customer ----------
st.header("2. Classify a New Customer")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=24)
    income = st.number_input(
        "Annual Income ($)",
        min_value=5000,
        max_value=300000,
        value=35000,
        step=1000
    )

with col2:
    spending = st.slider("Spending Score", 0, 100, 82)
    frequency = st.number_input(
        "Purchase Frequency / Month",
        min_value=0,
        max_value=30,
        value=8
    )

if st.button("🔍 Classify Customer", use_container_width=True):
    customer = [age, income, spending, frequency]
    prediction = system.classify_customer(customer)

    st.success(f"Predicted Segment: **Cluster {prediction['cluster']}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("KNN Confidence", f"{prediction['confidence'] * 100:.0f}%")
    c2.metric("Tie", "Yes" if prediction["tie"] else "No")
    c3.metric("KNN Votes", str(prediction["votes"]))

    st.subheader("Recommended Business Action")
    st.write(f"🎯 **{prediction['recommendation']}**")

# ---------- Technical evidence ----------
st.header("3. Reliability Evidence")

e1, e2 = st.columns(2)

with e1:
    st.metric("K-Means Inertia", f"{result['inertia']:.3f}")
    st.write("Lower inertia means points are closer to their assigned centroids.")

with e2:
    st.metric("Silhouette", f"{result['silhouette']:.3f}")
    st.write("A score above 0.5 indicates reasonably separated clusters.")

st.divider()
st.caption(
    "Core ML algorithms are implemented from scratch using Python lists "
    "and custom logic. Streamlit is used only for the application interface."
)
