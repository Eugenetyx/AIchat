"""
Smart Routing + SLA Demo
Streamlit UI only version (with round-robin routing)

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
if "round_robin_index" not in st.session_state:
    st.session_state.round_robin_index = 0
if "reroute_history" not in st.session_state:
    st.session_state.reroute_history = []

# --------------------------
# Routing Logic (Round Robin)
# --------------------------
def route_lead(lead_type: str):
    num_agents = len(AGENTS) - 1  # exclude AI agent
    attempts = 0

    while attempts < num_agents:
        agent = AGENTS[st.session_state.round_robin_index]["name"]
        st.session_state.round_robin_index = (st.session_state.round_robin_index + 1) % num_agents

        if len(st.session_state.agent_load[agent]) < AGENT_CAPACITY:
            return agent

        attempts += 1

    # All full → Cold → AI, Hot/Warm → least loaded human
    if lead_type == "Cold":
        return "AI Agent"

    human_loads = {a["name"]: len(st.session_state.agent_load[a["name"]]) for a in AGENTS if a["type"] != "ai"}
    return min(human_loads, key=human_loads.get)

# --------------------------
# Manual Status Update
# --------------------------
def update_lead_status(lead_id: str, new_status: str):
    for lead in st.session_state.leads:
        if lead["id"] == lead_id:
            old_status = lead["status"]
            lead["status"] = new_status
            lead["last_updated"] = datetime.now()
            
            # If changing to Success, remove from agent load
            if new_status == "Success":
                agent = lead["assigned_agent"]
                if lead_id in st.session_state.agent_load[agent]:
                    st.session_state.agent_load[agent].remove(lead_id)
            
            # If changing from Success back to Pending/Rerouted, add back to agent load
            elif old_status == "Success" and new_status in ["Pending", "Rerouted"]:
                agent = lead["assigned_agent"]
                if lead_id not in st.session_state.agent_load[agent]:
                    st.session_state.agent_load[agent].append(lead_id)
            
            break

# --------------------------
# SLA Monitoring with Reroute History
# --------------------------
def check_sla():
    now = datetime.now()
    for lead in st.session_state.leads:
        if lead["status"] == "Pending":
            mins_waited = (now - lead["timestamp"]).total_seconds() / 60
            if mins_waited > SLA_RULES[lead["type"]]:
                # Record reroute history
                old_agent = lead["assigned_agent"]
                old_agent_type = next(a["type"] for a in AGENTS if a["name"] == old_agent)
                
                # Remove from old agent
                if lead["id"] in st.session_state.agent_load[old_agent]:
                    st.session_state.agent_load[old_agent].remove(lead["id"])
                
                # Find new agent
                new_agent = route_lead(lead["type"])
                new_agent_type = next(a["type"] for a in AGENTS if a["name"] == new_agent)
                
                # Update lead
                lead["assigned_agent"] = new_agent
                lead["status"] = "Rerouted (SLA Breach)"
                st.session_state.agent_load[new_agent].append(lead["id"])
                
                # Log reroute reason
                reroute_record = {
                    "timestamp": now,
                    "lead_id": lead["id"],
                    "lead_name": lead["name"],
                    "lead_type": lead["type"],
                    "from_agent": old_agent,
                    "from_agent_type": old_agent_type,
                    "to_agent": new_agent,
                    "to_agent_type": new_agent_type,
                    "reason": f"SLA breach: waited {mins_waited:.1f} min, limit {SLA_RULES[lead['type']]} min",
                    "sla_limit": SLA_RULES[lead["type"]],
                    "actual_wait_time": mins_waited
                }
                st.session_state.reroute_history.append(reroute_record)

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
        "last_updated": datetime.now(),
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

    st.markdown(f"**🔄 Next Agent in Round-Robin:** {AGENTS[st.session_state.round_robin_index]['name']}")

    if st.session_state.leads:
        df = pd.DataFrame(st.session_state.leads)
        df["SLA Deadline (min)"] = df["type"].apply(lambda t: SLA_RULES[t])
        df["Time Since Submit (min)"] = df["timestamp"].apply(lambda t: round((datetime.now() - t).total_seconds() / 60, 1))
        df["SLA Breached"] = df["Time Since Submit (min)"] > df["SLA Deadline (min)"]

        # Manual Status Change Section
        st.subheader("📝 Manual Status Updates")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_lead = st.selectbox("Select Lead to Update:", 
                                       options=[f"{lead['id']} - {lead['name']}" for lead in st.session_state.leads],
                                       key="lead_selector")
        
        with col2:
            new_status = st.selectbox("New Status:", 
                                    options=["Pending", "Success", "Rerouted"],
                                    key="status_selector")
        
        with col3:
            if st.button("Update Status", key="update_button"):
                lead_id = selected_lead.split(" - ")[0]
                update_lead_status(lead_id, new_status)
                st.success(f"✅ Updated {lead_id} status to {new_status}")
                st.rerun()

        # Main leads table
        st.subheader("📋 Current Leads")
        
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
        # Reroute History & Logic Explanation
        # --------------------------
        st.subheader("🔄 Rerouting Logic & History")
        
        # Explanation of rerouting logic
        with st.expander("ℹ️ How Rerouting Works"):
            st.markdown("""
            **Automatic SLA-Based Rerouting Logic:**
            
            1. **SLA Monitoring**: System continuously checks if leads exceed their SLA time limits:
               - 🔥 **Hot leads**: 2 minutes
               - 🔸 **Warm leads**: 5 minutes  
               - ❄️ **Cold leads**: 15 minutes
            
            2. **When SLA is Breached**:
               - Lead is automatically removed from current agent's queue
               - System finds new agent using same routing logic
               - Status changes to "Rerouted (SLA Breach)"
               - Reroute event is logged with full details
            
            3. **Routing Priority**:
               - **Round-robin** through available agents first
               - If all human agents at capacity:
                 - **Cold leads** → Route to AI Agent
                 - **Hot/Warm leads** → Route to least loaded human agent
            
            4. **Agent Types**:
               - **Top Sales** (Sarah, John): Handle high-value leads
               - **Customer Service** (Amy, David, Lisa, Mike): General support
               - **AI Agent**: Handles overflow, mainly cold leads
            """)
        
        # Show reroute history if any
        if st.session_state.reroute_history:
            st.subheader("📊 Reroute History")
            reroute_df = pd.DataFrame(st.session_state.reroute_history)
            reroute_df["timestamp"] = reroute_df["timestamp"].dt.strftime("%H:%M:%S")
            
            # Display with enhanced formatting
            for _, row in reroute_df.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        st.write(f"⏰ {row['timestamp']}")
                    with col2:
                        st.write(f"**{row['lead_name']}** ({row['lead_type']} lead)")
                        st.write(f"🔄 {row['from_agent']} ({row['from_agent_type']}) → {row['to_agent']} ({row['to_agent_type']})")
                    with col3:
                        st.write(f"🚨 {row['reason']}")
                    st.divider()
        else:
            st.info("No rerouting has occurred yet.")

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
    agent_load_data = [{"Agent": a["name"], "Type": a["type"], "Current Leads": len(st.session_state.agent_load[a["name"]]), "Capacity": a["capacity"]} for a in AGENTS]
    st.table(pd.DataFrame(agent_load_data))

if __name__ == "__main__":
    run_app()
