"""
Smart Routing + SLA Demo
Streamlit UI only version (without external autorefresh dependency)

How to run locally:
  pip install streamlit pandas numpy plotly
  streamlit run main.py
"""

import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px

# --------------------------
# Agent Setup
# --------------------------
AGENT_CAPACITY = 5

AGENTS = [
    {"name": "Sarah", "type": "top_sales", "capacity": AGENT_CAPACITY},
    {"name": "John", "type": "top_sales", "capacity": AGENT_CAPACITY},
    {"name": "Amy", "type": "customer_service", "capacity": AGENT_CAPACITY},
    {"name": "David", "type": "customer_service", "capacity": AGENT_CAPACITY},
    {"name": "Lisa", "type": "customer_service", "capacity": AGENT_CAPACITY},
    {"name": "Mike", "type": "customer_service", "capacity": AGENT_CAPACITY},
    {"name": "AI Agent", "type": "ai", "capacity": 999},
]

SLA_RULES = {"Hot": 2, "Warm": 5, "Cold": 15}  # minutes SLA

# --------------------------
# State Initialization
# --------------------------
if "leads" not in st.session_state:
    st.session_state.leads = []
if "agent_load" not in st.session_state:
    st.session_state.agent_load = {agent["name"]: [] for agent in AGENTS}

# --------------------------
# Routing Logic
# --------------------------
def route_lead(lead_type: str):
    # Hot → Top Sales priority
    if lead_type == "Hot":
        for agent in ["Sarah", "John"]:
            if len(st.session_state.agent_load[agent]) < AGENT_CAPACITY:
                return agent

    # Try Customer Service
    for agent in ["Amy", "David", "Lisa", "Mike"]:
        if len(st.session_state.agent_load[agent]) < AGENT_CAPACITY:
            return agent

    # All full
    if lead_type == "Cold":
        return "AI Agent"

    # Warm/Hot → least loaded human
    human_loads = {a["name"]: len(st.session_state.agent_load[a["name"]]) for a in AGENTS if a["type"] != "ai"}
    return min(human_loads, key=human_loads.get)

# --------------------------
# SLA Monitoring
# --------------------------
def check_sla():
    now = datetime.now()
    for lead in st.session_state.leads:
        if lead["status"] == "Pending":
            mins_waited = (now - lead["timestamp"]).total_seconds() / 60
            if mins_waited > SLA_RULES[lead["type"]]:
                # Reroute
                old_agent = lead["assigned_agent"]
                if lead["id"] in st.session_state.agent_load[old_agent]:
                    st.session_state.agent_load[old_agent].remove(lead["id"])
                new_agent = route_lead(lead["type"])
                lead["assigned_agent"] = new_agent
                lead["status"] = "Rerouted (SLA Breach)"
                st.session_state.agent_load[new_agent].append(lead["id"])

# --------------------------
# Add Lead
# --------------------------
def add_lead(name: str, lead_type: str, alps_score: int = None):
    assigned_agent = route_lead(lead_type)
    lead_id = f"Lead_{len(st.session_state.leads) + 1}"
    new_lead = {
        "id": lead_id,
        "name": name,
        "type": lead_type,
        "alps_score": alps_score,
        "assigned_agent": assigned_agent,
        "timestamp": datetime.now(),
        "status": "Pending",
    }
    st.session_state.leads.append(new_lead)
    st.session_state.agent_load[assigned_agent].append(lead_id)

# --------------------------
# Realtime Simulation
# --------------------------
def simulate_realtime_lead():
    lead_type = random.choices(["Hot", "Warm", "Cold"], weights=[0.2, 0.5, 0.3])[0]
    name = f"Realtime_{len(st.session_state.leads) + 1}"
    alps_score = np.random.randint(40, 95)
    add_lead(name, lead_type, alps_score)

# --------------------------
# Streamlit UI
# --------------------------

def run_app():
    st.set_page_config(page_title="Smart Routing + SLA", layout="wide")

    # Sidebar
    st.sidebar.header("➕ Add Lead Manually")
    lead_name = st.sidebar.text_input("Customer Name")
    lead_type = st.sidebar.selectbox("Lead Type", ["Hot", "Warm", "Cold"])
    if st.sidebar.button("Submit Lead") and lead_name:
        add_lead(lead_name, lead_type)
        st.sidebar.success(f"✅ {lead_name} ({lead_type}) routed successfully!")

    if st.sidebar.button("Inject Random Lead Now"):
        simulate_realtime_lead()
        st.sidebar.info("📥 Random lead injected.")

    # SLA monitoring
    check_sla()

    # Dashboard
    st.title("🤖 Smart Routing + SLA Dashboard")

    if st.session_state.leads:
        df = pd.DataFrame(st.session_state.leads)
        df["SLA Deadline (min)"] = df["type"].apply(lambda t: SLA_RULES[t])
        df["Time Since Submit (min)"] = df["timestamp"].apply(lambda t: round((datetime.now() - t).total_seconds() / 60, 1))
        df["SLA Breached"] = df["Time Since Submit (min)"] > df["SLA Deadline (min)"]

        def highlight_sla(row):
            if row["SLA Breached"]:
                return ["background-color: red; color: white"] * len(row)
            elif row["Time Since Submit (min)"] > 0.8 * row["SLA Deadline (min)"]:
                return ["background-color: orange; color: black"] * len(row)
            else:
                return [""] * len(row)

        st.dataframe(df.style.apply(highlight_sla, axis=1), use_container_width=True)

        breached_count = int(df["SLA Breached"].sum())
        if breached_count > 0:
            st.error(f"🚨 {breached_count} leads have breached SLA! Please take action.")

        # --------------------------
        # Lead Distribution Chart
        # --------------------------
        st.subheader("📈 Lead Distribution")
        lead_counts = df["type"].value_counts().reset_index()
        lead_counts.columns = ["Lead Type", "Count"]
        fig = px.pie(lead_counts, names="Lead Type", values="Count", title="Hot vs Warm vs Cold Leads")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No leads yet. Add manually or wait for auto-injected leads.")

    # Agent load table
    st.subheader("📊 Agent Load Status")
    agent_load_data = [{"Agent": a["name"], "Current Leads": len(st.session_state.agent_load[a["name"]]), "Capacity": a["capacity"]} for a in AGENTS]
    st.table(pd.DataFrame(agent_load_data))

if __name__ == "__main__":
    run_app()
