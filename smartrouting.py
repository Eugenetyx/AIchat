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
def route_lead(lead_type: str, exclude_agent: str = None):
    """
    Route lead using round-robin, with option to exclude specific agent
    Args:
        lead_type: Hot/Warm/Cold lead type
        exclude_agent: Agent name to exclude from routing (for rerouting scenarios)
    """
    human_agents = [a for a in AGENTS if a["type"] != "ai"]
    num_agents = len(human_agents)
    attempts = 0
    
    # For new leads, use normal round-robin
    if exclude_agent is None:
        while attempts < num_agents:
            agent = AGENTS[st.session_state.round_robin_index]["name"]
            st.session_state.round_robin_index = (st.session_state.round_robin_index + 1) % num_agents

            if len(st.session_state.agent_load[agent]) < AGENT_CAPACITY:
                return agent
            attempts += 1
    
    # For rerouting: find best available agent excluding current one
    else:
        available_agents = []
        for agent in human_agents:
            agent_name = agent["name"]
            if agent_name != exclude_agent and len(st.session_state.agent_load[agent_name]) < AGENT_CAPACITY:
                available_agents.append((agent_name, len(st.session_state.agent_load[agent_name])))
        
        # If we have available agents, pick the least loaded one
        if available_agents:
            available_agents.sort(key=lambda x: x[1])  # Sort by load
            return available_agents[0][0]  # Return least loaded agent

    # Fallback: All human agents full or excluded
    if lead_type == "Cold":
        return "AI Agent"
    
    # For Hot/Warm: find least loaded human (even if over capacity)
    human_loads = {a["name"]: len(st.session_state.agent_load[a["name"]]) 
                   for a in human_agents if a["name"] != exclude_agent}
    
    if human_loads:
        return min(human_loads, key=human_loads.get)
    else:
        # Edge case: all humans excluded, route to AI
        return "AI Agent"

# --------------------------
# Manual Status Update
# --------------------------
def update_lead_status(lead_id: str, new_status: str):
    for lead in st.session_state.leads:
        if lead["id"] == lead_id:
            old_status = lead["status"]
            lead["status"] = new_status
            current_time = datetime.now()
            lead["last_updated"] = current_time
            
            # Reset timing based on status change
            if new_status == "Success":
                # Record completion time and freeze timer
                lead["completion_time"] = current_time
                lead["time_to_complete"] = (current_time - lead["timestamp"]).total_seconds() / 60
                agent = lead["assigned_agent"]
                if lead_id in st.session_state.agent_load[agent]:
                    st.session_state.agent_load[agent].remove(lead_id)
            
            elif new_status == "Rerouted":
                # Reset timer for rerouted leads
                lead["reroute_timestamp"] = current_time
                if old_status == "Success":
                    # If changing from Success back to Rerouted, add back to agent load
                    agent = lead["assigned_agent"]
                    if lead_id not in st.session_state.agent_load[agent]:
                        st.session_state.agent_load[agent].append(lead_id)
            
            elif new_status == "Pending":
                if old_status == "Success":
                    # If changing from Success back to Pending, add back to agent load
                    agent = lead["assigned_agent"]
                    if lead_id not in st.session_state.agent_load[agent]:
                        st.session_state.agent_load[agent].append(lead_id)
                elif old_status.startswith("Rerouted"):
                    # Reset timer when changing from Rerouted to Pending
                    lead["reroute_timestamp"] = current_time
            
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
                
                # Find new agent (exclude current agent to prevent routing back)
                new_agent = route_lead(lead["type"], exclude_agent=old_agent)
                new_agent_type = next(a["type"] for a in AGENTS if a["name"] == new_agent)
                
                # Update lead
                lead["assigned_agent"] = new_agent
                lead["status"] = "Rerouted (SLA Breach)"
                lead["reroute_timestamp"] = now  # Reset timer for rerouted lead
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
# Score-based Lead Type Classification
# --------------------------
def determine_lead_type(alps_score: int) -> str:
    """Determine lead type based on ALPS score"""
    if alps_score >= 71:
        return "Hot"
    elif alps_score >= 51:
        return "Warm"
    else:
        return "Cold"

# --------------------------
# Add Lead
# --------------------------
def add_lead(name: str, alps_score: int, lead_type: str = None):
    # If lead_type not provided, determine from score
    if lead_type is None:
        lead_type = determine_lead_type(alps_score)
    
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
    alps_score = np.random.randint(0, 101)  # 0-100 score range
    lead_type = determine_lead_type(alps_score)
    name = f"Realtime_{len(st.session_state.leads) + 1}"
    add_lead(name, alps_score)

# --------------------------
# Streamlit UI
# --------------------------

def run_app():
    st.set_page_config(page_title="Smart Routing + SLA", layout="wide")

    # Sidebar
    st.sidebar.header("➕ Add Lead Manually")
    lead_name = st.sidebar.text_input("Customer Name")
    alps_score = st.sidebar.number_input("ALPS Score (0-100)", min_value=0, max_value=100, value=50)
    
    # Show predicted lead type based on score
    predicted_type = determine_lead_type(alps_score)
    st.sidebar.info(f"📊 Predicted Lead Type: **{predicted_type}**")
    st.sidebar.caption("Cold: 0-50 | Warm: 51-70 | Hot: 71-100")
    
    if st.sidebar.button("Submit Lead") and lead_name:
        add_lead(lead_name, alps_score)
        st.sidebar.success(f"✅ {lead_name} (Score: {alps_score}, Type: {predicted_type}) routed successfully!")

    if st.sidebar.button("Inject Random Lead Now"):
        simulate_realtime_lead()
        last_lead = st.session_state.leads[-1]
        st.sidebar.info(f"📥 Random lead injected: Score {last_lead['alps_score']} ({last_lead['type']})")

    # SLA monitoring
    check_sla()

    # Dashboard
    st.title("🤖 Smart Routing + SLA Dashboard")

    st.markdown(f"**🔄 Next Agent in Round-Robin:** {AGENTS[st.session_state.round_robin_index]['name']}")

    if st.session_state.leads:
        df = pd.DataFrame(st.session_state.leads)
        df["SLA Deadline (min)"] = df["type"].apply(lambda t: SLA_RULES[t])
        
        # Calculate time based on status
        def calculate_time_since_submit(row):
            current_time = datetime.now()
            
            if row["status"] == "Success":
                # Show completion time (frozen)
                if "time_to_complete" in row and pd.notna(row["time_to_complete"]):
                    return round(row["time_to_complete"], 1)
                else:
                    # Fallback for old success records
                    return round((row.get("completion_time", current_time) - row["timestamp"]).total_seconds() / 60, 1)
            
            elif row["status"].startswith("Rerouted"):
                # Time since reroute (reset timer)
                if "reroute_timestamp" in row and pd.notna(row["reroute_timestamp"]):
                    return round((current_time - row["reroute_timestamp"]).total_seconds() / 60, 1)
                else:
                    # Fallback: time since last update
                    return round((current_time - row["last_updated"]).total_seconds() / 60, 1)
            
            else:  # Pending status
                # Normal time since original submission
                return round((current_time - row["timestamp"]).total_seconds() / 60, 1)
        
        df["Time Since Submit (min)"] = df.apply(calculate_time_since_submit, axis=1)
        df["SLA Breached"] = (df["Time Since Submit (min)"] > df["SLA Deadline (min)"]) & (df["status"] != "Success")

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

        # Main leads table with refresh button
        st.subheader("📋 Current Leads")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Total Leads:** {len(df)} | **Pending:** {len(df[df['status']=='Pending'])} | **Success:** {len(df[df['status']=='Success'])} | **Rerouted:** {len(df[df['status'].str.contains('Rerouted', na=False)])}")
        with col2:
            if st.button("🔄 Refresh Dashboard", key="refresh_main"):
                st.rerun()
        
        def highlight_rows(row):
            """Apply color coding based on status and SLA"""
            if row["status"] == "Success":
                return ["background-color: #d4edda; color: #155724"] * len(row)  # Green
            elif row["status"].startswith("Rerouted"):
                return ["background-color: #fff3cd; color: #856404"] * len(row)  # Yellow
            elif row["status"] != "Success" and row["SLA Breached"]:
                return ["background-color: #f8d7da; color: #721c24"] * len(row)  # Red
            elif row["status"] != "Success" and row["Time Since Submit (min)"] > 0.8 * row["SLA Deadline (min)"]:
                return ["background-color: #ffeaa7; color: #856404"] * len(row)  # Orange warning
            else:
                return [""] * len(row)  # Default

        # Display table with enhanced styling
        styled_df = df.style.apply(highlight_rows, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Color legend
        st.markdown("""
        **📊 Status Legend & Timer Logic:**
        - 🟢 **Green**: Success (timer shows completion time - frozen)
        - 🟡 **Yellow**: Rerouted (timer resets from reroute moment)  
        - 🟠 **Orange**: Warning (approaching SLA limit)
        - 🔴 **Red**: SLA breached (immediate attention needed)
        
        **⏱️ Timer Behavior:**
        - **Success**: Shows total time to complete (frozen)
        - **Rerouted**: Shows time since last reroute (reset)
        - **Pending**: Shows time since original submission
        """)

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
               - System finds new agent **excluding current agent** (prevents routing back)
               - **Priority**: Least loaded available agent first
               - Status changes to "Rerouted (SLA Breach)"
               - Reroute event is logged with full details
            
            3. **Routing Priority for Rerouting**:
               - **First**: Try least loaded human agent (excluding current)
               - **If no humans available**:
                 - **Cold leads** → Route to AI Agent
                 - **Hot/Warm leads** → Route to least loaded human (even if over capacity)
            
            4. **New Lead Routing** (Round-Robin):
               - Cycles through agents: Sarah → John → Amy → David → Lisa → Mike → repeat
               - Only assigns if agent has capacity (< 5 leads)
               - If all full: Cold→AI, Hot/Warm→least loaded
            
            5. **Score-Based Lead Classification**:
               - 🔥 **Hot leads**: ALPS Score 71-100 (High conversion potential)
               - 🔸 **Warm leads**: ALPS Score 51-70 (Medium conversion potential)
               - ❄️ **Cold leads**: ALPS Score 0-50 (Low conversion potential)
               - Lead type automatically determined by score
            
            6. **Agent Types**:
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
