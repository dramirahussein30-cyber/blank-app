import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# ==========================================
# 0. Optional: Mapbox token (set in Streamlit secrets)
# ==========================================
MAPBOX_TOKEN = st.secrets.get("MAPBOX_TOKEN", None)
if MAPBOX_TOKEN:
    px.set_mapbox_access_token(MAPBOX_TOKEN)

# ==========================================
# 1. ENTERPRISE UI CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="PharmaFlow AI | Strategic Command Center",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Enterprise Feel
st.markdown(
    """
<style>
    .stApp { background-color: #f8f9fa; }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4e8cff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Playbook Cards */
    .playbook-card {
        background-color: #eef2ff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #c7d2fe;
        margin-bottom: 10px;
    }
    
    /* Headers */
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; color: #1e293b; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0f172a; color: white; }
    
    /* Buttons */
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 6px;
        font-weight: 600;
        border: none;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 2. CONTEXT-AWARE DATA GENERATION
# ==========================================
@st.cache_data
def load_strategic_data(seed: int = 42) -> pd.DataFrame:
    """Generates data enriched with Egypt-Specific Intelligence."""
    np.random.seed(seed)

    regions = ["North (Cairo)", "West (Alex)", "East (Canal)", "South (Upper Egypt)"]
    segments = ["Platinum", "Gold", "Silver", "Bronze"]

    data = []
    for i in range(400):
        reg = np.random.choice(regions, p=[0.35, 0.25, 0.2, 0.2])
        rev = np.random.randint(10000, 500000)

        # Segment Logic
        if rev > 250000:
            seg = "Platinum"
        elif rev > 100000:
            seg = "Gold"
        elif rev > 40000:
            seg = "Silver"
        else:
            seg = "Bronze"

        # Egypt Context Features
        route_delay_risk = "High" if reg == "South (Upper Egypt)" else "Low"
        payment_risk = np.random.choice(["Low", "Medium", "High"], p=[0.6, 0.3, 0.1])
        ramadan_index = np.random.uniform(1.2, 1.8)  # 1.2x to 1.8x demand surge during Ramadan

        # Lat/Lon with Clustering
        if "North" in reg:
            lat, lon = 30.0444, 31.2357
        elif "West" in reg:
            lat, lon = 31.2001, 29.9187
        elif "East" in reg:
            lat, lon = 30.5852, 32.2654
        else:
            lat, lon = 26.1551, 32.7160

        # Churn Explainability
        churn_score = np.random.uniform(0, 1)
        churn_reasons = []
        if churn_score > 0.6:
            churn_reasons = np.random.choice(
                ["Late Delivery", "Price Sensitivity", "Competitor Promo", "Stockouts"],
                size=2,
                replace=False,
            )

        data.append(
            {
                "ID": f"PH-{1000+i}",
                "Name": f"Pharmacy {i}",
                "Region": reg,
                "Segment": seg,
                "Revenue": rev,
                "Churn Risk": churn_score,
                "Churn Drivers": ", ".join(churn_reasons) if len(churn_reasons) > 0 else "None",
                "Ramadan Index": ramadan_index,
                "Route Delay Risk": route_delay_risk,
                "Payment Risk": payment_risk,
                "Lat": lat + np.random.normal(0, 0.5),
                "Lon": lon + np.random.normal(0, 0.5),
                "Last Visit": np.random.randint(1, 45),
            }
        )

    return pd.DataFrame(data)


df = load_strategic_data()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.title("🧬 PharmaFlow AI")
    st.caption("Strategic Enterprise Edition")
    st.markdown("---")

    menu = st.radio(
        "STRATEGIC MODULES",
        [
            "🚀 Executive Command",
            "🧠 Egypt Intelligence Layer",
            "💎 Pharmacy 360 & Playbooks",
            "🗺️ Territory & Route Opt.",
            "💰 Monetization & Suppliers",
            "🤖 Autonomous Procurement",
            "⚖️ Governance & Audit",
        ],
    )

    st.markdown("---")
    st.info("System Online\nModel Accuracy: 91.4%")

# ==========================================
# 4. MODULE: EXECUTIVE COMMAND
# ==========================================
if menu == "🚀 Executive Command":
    st.title("🚀 Executive Command Center")
    st.markdown("High-level operational oversight.")

    # Top Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Monthly Revenue", f"EGP {df['Revenue'].sum() / 1_000_000:.1f}M", "+12%")
    c2.metric("High Churn Risk", int(len(df[df["Churn Risk"] > 0.7])), "-5 vs Last Week")
    c3.metric("Avg. Ramadan Lift", "1.45x", "Predicted")
    c4.metric("Monetized Data Revenue", "EGP 250k", "New Stream")

    st.markdown("---")

    col_map, col_pie = st.columns([2, 1])
    with col_map:
        st.subheader("🌍 Regional Risk Heatmap")
        fig_map = px.scatter_mapbox(
            df,
            lat="Lat",
            lon="Lon",
            color="Churn Risk",
            size="Revenue",
            color_continuous_scale="RdYlGn_r",
            zoom=5,
            height=500,
            mapbox_style="carto-positron",
            hover_name="Name",
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_pie:
        st.subheader("📊 Segment Distribution")
        fig_pie = px.pie(df, names="Segment", values="Revenue", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# 5. MODULE: EGYPT INTELLIGENCE LAYER (New)
# ==========================================
elif menu == "🧠 Egypt Intelligence Layer":
    st.title("🧠 Context-Aware Feature Layer")
    st.markdown("The 'Secret Sauce': Egypt-Specific Market Intelligence.")

    tab1, tab2, tab3 = st.tabs(["🌙 Ramadan Spikes", "🚚 Route & Access", "💊 Seasonality Profiles"])

    with tab1:
        st.subheader("🌙 Ramadan Consumption Index")
        st.caption("Predictive multiplier for consumption changes during Holy Month.")

        # Visualization of Demand Shift
        hours = list(range(24))
        normal_demand = [10, 8, 5, 2, 1, 1, 2, 5, 10, 15, 20, 25, 25, 20, 15, 15, 20, 30, 40, 35, 25, 15, 10, 8]
        ramadan_demand = [30, 25, 40, 10, 2, 1, 1, 2, 5, 8, 10, 15, 15, 20, 30, 45, 60, 50, 20, 15, 30, 40, 35, 30]

        fig_ram = go.Figure()
        fig_ram.add_trace(
            go.Scatter(
                x=hours,
                y=normal_demand,
                name="Normal Baseline",
                line=dict(color="gray", dash="dash"),
            )
        )
        fig_ram.add_trace(
            go.Scatter(
                x=hours,
                y=ramadan_demand,
                name="Ramadan Forecast",
                line=dict(color="#4e8cff", width=3),
            )
        )
        fig_ram.update_layout(
            title="Hourly Demand Shift (Iftar/Suhoor Effect)",
            xaxis_title="Hour of Day",
            yaxis_title="Order Volume Index",
        )
        st.plotly_chart(fig_ram, use_container_width=True)

    with tab2:
        st.subheader("🚚 Geographic Access Constraints")
        col_a, col_b = st.columns(2)
        with col_a:
            delay_counts = df["Route Delay Risk"].value_counts()
            fig_bar = px.bar(
                x=delay_counts.index,
                y=delay_counts.values,
                labels={"x": "Route Delay Risk", "y": "Count"},
                title="Route Delay Risk by Pharmacy Count",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_b:
            st.info(
                "💡 **Insight:** 'South Region' shows 60% High Delay Risk due to road infrastructure. **Action:** Buffer stock levels in Assiut Hub increased by 15%."
            )

    with tab3:
        st.subheader("🌡️ Therapeutic Seasonality")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        cats = ["Antibiotics", "Vitamins", "Skincare", "Chronic", "Cold/Flu"]
        z_data = [
            [90, 85, 70, 60, 50, 40, 30, 30, 40, 60, 80, 95],  # Antibiotics (Winter peak)
            [60, 60, 70, 80, 80, 70, 60, 60, 70, 80, 90, 80],  # Vitamins
            [30, 40, 50, 70, 90, 100, 100, 90, 70, 50, 40, 30],  # Skincare (Summer peak)
            [80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80],  # Chronic (Stable)
            [95, 80, 50, 20, 10, 5, 5, 10, 30, 60, 85, 100],  # Cold/Flu
        ]
        fig_heat = px.imshow(z_data, x=months, y=cats, color_continuous_scale="Blues", title="Category Seasonality Matrix")
        st.plotly_chart(fig_heat, use_container_width=True)

# ==========================================
# 6. MODULE: PHARMACY 360 & PLAYBOOKS
# ==========================================
elif menu == "💎 Pharmacy 360 & Playbooks":
    st.title("💎 Pharmacy 360° & Commercial Playbooks")

    # Selection
    selected_ph = st.selectbox("Search Pharmacy:", df["Name"].unique())
    p_rows = df[df["Name"] == selected_ph]
    if p_rows.empty:
        st.error("Selected pharmacy not found.")
    else:
        p_data = p_rows.iloc[0]

        # --- Profile Header ---
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            st.image("https://cdn-icons-png.flaticon.com/512/3063/3063823.png", width=100)
        with c2:
            st.subheader(f"{p_data['Name']} ({p_data['ID']})")
            st.caption(f"📍 {p_data['Region']} | Segment: **{p_data['Segment']}**")
            churn_pct = int(np.clip(p_data["Churn Risk"] * 100, 0, 100))
            st.progress(churn_pct)
            st.caption(f"Churn Risk: {churn_pct}%")
        with c3:
            st.metric("Ramadan Potential", f"{p_data['Ramadan Index']:.2f}x", "Lift")

        st.markdown("---")

        # --- Explainability & Trust Layer ---
        col_x, col_y = st.columns(2)

        with col_x:
            st.subheader("🔍 Explainability Layer")
            st.markdown("**Why is the Churn Score high?**")
            if p_data["Churn Risk"] > 0.5:
                drivers = p_data["Churn Drivers"].split(", ")
                for d in drivers:
                    if d and d != "None":
                        st.warning(f"⚠ **{d}**: Impact High")
                st.caption("Confidence Level: **High (92%)**")
            else:
                st.success("✅ Customer is stable. Key driver: Consistent Payment History.")

        with col_y:
            st.subheader("📘 AI Commercial Playbook")

            # Dynamic Playbook Logic
            if p_data["Segment"] == "Platinum":
                st.markdown(
                    """
                <div class="playbook-card">
                    <h4>🤝 Partnership Playbook</h4>
                    <ul>
                        <li><b>Strategy:</b> Lock-in annual contract.</li>
                        <li><b>Offer:</b> 2% Rebate for volume commitment.</li>
                        <li><b>Action:</b> Schedule Quarterly Business Review (QBR).</li>
                    </ul>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            elif p_data["Churn Risk"] > 0.6:
                st.markdown(
                    """
                <div class="playbook-card">
                    <h4>🛡️ Retention Playbook</h4>
                    <ul>
                        <li><b>Strategy:</b> Service Recovery.</li>
                        <li><b>Offer:</b> Free 'Express Delivery' for next 3 orders.</li>
                        <li><b>Action:</b> Sales Rep visit within 48h.</li>
                    </ul>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:  # Growth
                st.markdown(
                    """
                <div class="playbook-card">
                    <h4>📈 Growth Playbook</h4>
                    <ul>
                        <li><b>Strategy:</b> Category Expansion.</li>
                        <li><b>Offer:</b> 'Diabetes Bundle' (Cross-sell Strips).</li>
                        <li><b>Action:</b> Push notification via App.</li>
                    </ul>
                </div>
                """,
                    unsafe_allow_html=True,
                )

# ==========================================
# 7. MODULE: TERRITORY & ROUTE OPT (New)
# ==========================================
elif menu == "🗺️ Territory & Route Opt.":
    st.title("🗺️ Territory Optimization & Routing")
    st.markdown("Turns AI into field execution improvements.")

    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("📍 Risk/Revenue Priority Map")
        # Filter for high priority
        priority_df = df[(df["Revenue"] > 200_000) | (df["Churn Risk"] > 0.7)]

        fig_route = px.scatter_mapbox(
            priority_df,
            lat="Lat",
            lon="Lon",
            color="Churn Risk",
            size="Revenue",
            color_continuous_scale="Reds",
            zoom=6,
            height=500,
            mapbox_style="carto-positron",
            hover_name="Name",
            title="High-Value & At-Risk Targets",
        )
        st.plotly_chart(fig_route, use_container_width=True)

    with c2:
        st.subheader("🚗 Suggested Daily Route")
        st.caption("Optimized for Rep: Hassan (Alexandria)")

        route_list = priority_df[priority_df["Region"] == "West (Alex)"].head(5)

        for i, row in route_list.reset_index().iterrows():
            st.markdown(f"**{i+1}. {row['Name']}**")
            st.caption(f"Dist: {2*i + 1.5} km | Priority: {'🔴 High' if row['Churn Risk']>0.7 else '🟢 Growth'}")
            st.markdown("---")

        if st.button("📲 Push Route to Rep App"):
            # Provide simple feedback; in production push to an API / message queue
            st.success("Route sent to Rep's tablet!")

# ==========================================
# 8. MODULE: MONETIZATION (New)
# ==========================================
elif menu == "💰 Monetization & Suppliers":
    st.title("💰 Monetizable Add-Ons")
    st.markdown("Revenue generation from Data & Supplier Services.")

    tab1, tab2 = st.tabs(["🏭 Supplier Intelligence", "🏷️ Dynamic Pricing"])

    with tab1:
        st.subheader("Supplier Insights Dashboard (Paid Service)")
        st.markdown("*View sold to Manufacturers (e.g., Pfizer, Sanofi)*")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Your Market Share (Region)", "32%", "+2%")
        with c2:
            st.metric("Competitor Substitution Rate", "12%", "High Risk")

        # Regional Demand Heatmap
        st.subheader("Regional Demand Heatmap (Antibiotics)")
        fig_heat = px.density_mapbox(
            df,
            lat="Lat",
            lon="Lon",
            z="Revenue",
            radius=20,
            zoom=5,
            mapbox_style="carto-positron",
            height=400,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab2:
        st.subheader("🏷️ AI Dynamic Pricing Engine")
        st.markdown("Optimizes discounts to maximize margin vs. conversion.")

        col_x, col_y = st.columns(2)
        with col_x:
            base_price = 100
            discount = st.slider("Test Discount Level %", 0, 30, 10)

            # Simulated Elasticity
            conversion = 5 + (discount * 1.5)
            margin = (base_price * (1 - discount / 100)) - 60  # Cost 60
            profit = conversion * margin

            st.metric("Predicted Conversion", f"{conversion:.1f}%")
            st.metric("Projected Profit Index", f"{profit:.0f}")

        with col_y:
            st.info("💡 **AI Recommendation:** A **12% Discount** maximizes total profit for 'Gold Cluster' on 'Vitamins'.")

# ==========================================
# 9. MODULE: AUTONOMOUS PROCUREMENT
# ==========================================
elif menu == "🤖 Autonomous Procurement":
    st.title("🤖 Autonomous Procurement Agent (Phase 3)")
    st.info("Status: **Active** for Low-Risk SKUs")

    st.subheader("🛒 Auto-Replenishment Queue")

    proc_df = pd.DataFrame(
        {
            "SKU": ["Panadol Extra", "Insulin Lantus", "Baby Formula"],
            "Supplier": ["PharmaOverseas", "United Pharma", "Ibnsina"],
            "Qty": [5000, 200, 1000],
            "Confidence": ["99%", "95%", "92%"],
            "Status": ["⏳ Auto-Ordering in 10m", "⏳ Auto-Ordering in 45m", "⚠ Needs Approval"],
        }
    )

    st.dataframe(proc_df, use_container_width=True)

    # Use session_state to track agent state
    if "agent_paused" not in st.session_state:
        st.session_state["agent_paused"] = False

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⛔ Pause Agent"):
            st.session_state["agent_paused"] = True
            st.warning("Agent Paused.")
    with c2:
        if st.button("✅ Force Execute All"):
            # In production, call order-execution API here
            st.success("Orders Placed Successfully.")

    if st.session_state["agent_paused"]:
        st.info("Agent is currently paused. Use controls to resume.")

# ==========================================
# 10. MODULE: GOVERNANCE
# ==========================================
elif menu == "⚖️ Governance & Audit":
    st.title("⚖️ Governance & Ethical AI")

    st.metric("Fairness Score (South Region)", "98/100", "Compliant")

    st.subheader("📜 DPO Audit Log")
    log_df = pd.DataFrame(
        {
            "Time": ["10:00 AM", "Yesterday"],
            "User": ["System_Agent", "Sales_Admin"],
            "Action": ["Auto-Order Placed: Panadol", "Exported Competitor Intel Report"],
        }
    )
    st.dataframe(log_df, use_container_width=True)
