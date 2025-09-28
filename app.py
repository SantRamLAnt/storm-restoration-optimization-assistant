import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import networkx as nx
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple
import json

# Set page config
st.set_page_config(
    page_title="Storm Restoration Optimization Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .priority-high { color: #ff4757; font-weight: bold; }
    .priority-medium { color: #ffa502; font-weight: bold; }
    .priority-low { color: #2ed573; font-weight: bold; }
    .crew-available { color: #2ed573; }
    .crew-deployed { color: #ff6b6b; }
    .crew-returning { color: #ffa502; }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
</style>
""", unsafe_allow_html=True)

class StormRestorationOptimizer:
    def __init__(self):
        self.initialize_data()
    
    def initialize_data(self):
        """Initialize simulation data for storm restoration scenario"""
        # Set random seed for reproducible results
        np.random.seed(42)
        random.seed(42)
        
        # Generate outage locations
        self.outages = self.generate_outages()
        self.crews = self.generate_crews()
        self.materials = self.generate_materials()
        self.optimization_results = self.run_optimization()
        
    def generate_outages(self) -> pd.DataFrame:
        """Generate realistic outage data for storm scenario"""
        n_outages = 45
        
        # Miami-Dade area coordinates
        base_lat, base_lon = 25.7617, -80.1918
        
        outages = []
        for i in range(n_outages):
            outage = {
                'outage_id': f'OUT_{i+1:03d}',
                'latitude': base_lat + np.random.normal(0, 0.2),
                'longitude': base_lon + np.random.normal(0, 0.3),
                'customers_affected': np.random.randint(50, 2500),
                'priority': np.random.choice(['High', 'Medium', 'Low'], p=[0.3, 0.5, 0.2]),
                'equipment_type': np.random.choice(['Transformer', 'Distribution Line', 'Substation', 'Switch'], 
                                                 p=[0.4, 0.35, 0.15, 0.1]),
                'estimated_repair_time': np.random.randint(2, 12),
                'materials_needed': np.random.choice(['Poles', 'Wire', 'Transformer', 'Switches'], 
                                                   p=[0.3, 0.25, 0.25, 0.2]),
                'accessibility': np.random.choice(['Easy', 'Moderate', 'Difficult'], p=[0.4, 0.4, 0.2]),
                'report_time': datetime.now() - timedelta(hours=np.random.randint(1, 8)),
                'status': np.random.choice(['Pending', 'In Progress', 'Completed'], p=[0.6, 0.3, 0.1])
            }
            outages.append(outage)
        
        return pd.DataFrame(outages)
    
    def generate_crews(self) -> pd.DataFrame:
        """Generate crew data with skills and locations"""
        n_crews = 15
        
        crews = []
        for i in range(n_crews):
            crew = {
                'crew_id': f'CREW_{i+1:02d}',
                'latitude': 25.7617 + np.random.normal(0, 0.15),
                'longitude': -80.1918 + np.random.normal(0, 0.25),
                'status': np.random.choice(['Available', 'Deployed', 'Returning'], p=[0.5, 0.4, 0.1]),
                'skill_level': np.random.choice(['Standard', 'Advanced', 'Specialist'], p=[0.5, 0.3, 0.2]),
                'equipment': np.random.choice(['Bucket Truck', 'Line Truck', 'Emergency Vehicle'], p=[0.4, 0.4, 0.2]),
                'current_assignment': np.random.choice([None, 'OUT_001', 'OUT_015', 'OUT_032'], p=[0.5, 0.2, 0.2, 0.1]),
                'estimated_completion': datetime.now() + timedelta(hours=np.random.randint(1, 6)) if np.random.random() > 0.5 else None,
                'work_hours_today': np.random.randint(2, 10)
            }
            crews.append(crew)
        
        return pd.DataFrame(crews)
    
    def generate_materials(self) -> pd.DataFrame:
        """Generate materials inventory data"""
        materials = [
            {'material': 'Distribution Poles', 'available': 150, 'required': 45, 'location': 'Warehouse A'},
            {'material': 'Primary Wire (ft)', 'available': 12500, 'required': 8200, 'location': 'Warehouse A'},
            {'material': 'Transformers', 'available': 25, 'required': 18, 'location': 'Warehouse B'},
            {'material': 'Switches', 'available': 40, 'required': 15, 'location': 'Warehouse A'},
            {'material': 'Insulators', 'available': 200, 'required': 95, 'location': 'Warehouse C'},
            {'material': 'Cross Arms', 'available': 80, 'required': 35, 'location': 'Warehouse B'}
        ]
        
        return pd.DataFrame(materials)
    
    def run_optimization(self) -> Dict:
        """Simulate optimization engine results"""
        # Calculate optimization metrics
        total_customers = self.outages['customers_affected'].sum()
        pending_outages = len(self.outages[self.outages['status'] == 'Pending'])
        available_crews = len(self.crews[self.crews['status'] == 'Available'])
        
        # Simulate optimization results
        results = {
            'total_outages': len(self.outages),
            'pending_outages': pending_outages,
            'customers_affected': total_customers,
            'available_crews': available_crews,
            'estimated_restoration_time': '14.5 hours',
            'optimization_score': 87.3,
            'resource_utilization': 92.1,
            'travel_time_reduction': '45%',
            'priority_completion_rate': 89.2
        }
        
        return results

def main():
    # Initialize optimizer
    optimizer = StormRestorationOptimizer()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚡ Storm Restoration Optimization Assistant</h1>
        <p>Accelerated Emergency Response Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced Demo Banner
    st.info("🚀 **Advanced Demo:** This platform showcases AI-driven crew routing and logistics optimization post-storm using real-time outage clustering, PyTorch Geometric GNNs, Gurobi optimization, and mobile dispatch integration.")
    
    # Sidebar Controls
    st.sidebar.header("🎛️ Analysis Controls")
    
    # Storm scenario selection
    scenario = st.sidebar.selectbox(
        "Storm Scenario",
        ["Hurricane Category 2", "Severe Thunderstorm", "Ice Storm", "Tornado Outbreak"],
        index=0
    )
    
    # Optimization parameters
    st.sidebar.subheader("Optimization Parameters")
    max_travel_time = st.sidebar.slider("Max Travel Time (hours)", 0.5, 3.0, 1.5, 0.1)
    crew_capacity = st.sidebar.slider("Crew Capacity Factor", 0.5, 1.5, 1.0, 0.1)
    priority_weight = st.sidebar.slider("Priority Weight", 1.0, 5.0, 2.5, 0.1)
    
    # Real-time updates toggle
    real_time = st.sidebar.checkbox("Real-time Updates", value=True)
    
    if real_time:
        st.sidebar.success("🔄 Live data streaming active")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ACTIVE OUTAGES</div>
            <div class="metric-value">{optimizer.optimization_results['total_outages']}</div>
            <div class="metric-label">Storm Impact Event</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CUSTOMERS AFFECTED</div>
            <div class="metric-value">{optimizer.optimization_results['customers_affected']:,}</div>
            <div class="metric-label">Awaiting Restoration</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">AVAILABLE CREWS</div>
            <div class="metric-value">{optimizer.optimization_results['available_crews']}</div>
            <div class="metric-label">Ready for Dispatch</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">EST. RESTORATION</div>
            <div class="metric-value">{optimizer.optimization_results['estimated_restoration_time']}</div>
            <div class="metric-label">AI-Optimized Timeline</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Outage Clustering", 
        "🚛 Crew Optimization", 
        "📦 Resource Planning", 
        "📊 Performance Analytics"
    ])
    
    with tab1:
        st.subheader("🎯 Real-time Outage Clustering & Prioritization")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create outage map
            m = folium.Map(
                location=[25.7617, -80.1918],
                zoom_start=10,
                tiles='OpenStreetMap'
            )
            
            # Add outages to map
            for _, outage in optimizer.outages.iterrows():
                color = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}[outage['priority']]
                size = max(5, min(20, outage['customers_affected'] / 100))
                
                folium.CircleMarker(
                    location=[outage['latitude'], outage['longitude']],
                    radius=size,
                    popup=f"""
                    <b>{outage['outage_id']}</b><br>
                    Priority: {outage['priority']}<br>
                    Customers: {outage['customers_affected']:,}<br>
                    Equipment: {outage['equipment_type']}<br>
                    Est. Time: {outage['estimated_repair_time']}h
                    """,
                    color=color,
                    fillColor=color,
                    fillOpacity=0.6
                ).add_to(m)
            
            # Add crew locations
            for _, crew in optimizer.crews.iterrows():
                icon_color = {'Available': 'green', 'Deployed': 'red', 'Returning': 'orange'}[crew['status']]
                
                folium.Marker(
                    location=[crew['latitude'], crew['longitude']],
                    popup=f"""
                    <b>{crew['crew_id']}</b><br>
                    Status: {crew['status']}<br>
                    Skill: {crew['skill_level']}<br>
                    Equipment: {crew['equipment']}
                    """,
                    icon=folium.Icon(color=icon_color, icon='wrench', prefix='fa')
                ).add_to(m)
            
            st_folium(m, width=700, height=500)
        
        with col2:
            st.markdown("#### Outage Summary")
            
            # Priority distribution
            priority_counts = optimizer.outages['priority'].value_counts()
            fig_priority = px.pie(
                values=priority_counts.values,
                names=priority_counts.index,
                title="Outages by Priority",
                color_discrete_map={'High': '#ff4757', 'Medium': '#ffa502', 'Low': '#2ed573'}
            )
            fig_priority.update_layout(height=250, showlegend=True)
            st.plotly_chart(fig_priority, use_container_width=True)
            
            # Equipment type distribution
            equipment_counts = optimizer.outages['equipment_type'].value_counts()
            fig_equipment = px.bar(
                x=equipment_counts.values,
                y=equipment_counts.index,
                orientation='h',
                title="Equipment Types",
                color=equipment_counts.values,
                color_continuous_scale='Viridis'
            )
            fig_equipment.update_layout(height=250)
            st.plotly_chart(fig_equipment, use_container_width=True)
        
        # Outage clustering analysis
        st.markdown("#### 🧠 Graph Neural Network Clustering Results")
        
        # Simulate GNN clustering results
        cluster_data = pd.DataFrame({
            'Cluster': ['North Miami', 'Downtown', 'Coral Gables', 'Homestead', 'Key Biscayne'],
            'Outages': [12, 8, 7, 11, 7],
            'Customers': [15420, 9850, 6320, 18900, 4180],
            'Avg Priority Score': [8.2, 9.1, 6.5, 7.8, 5.9],
            'Est. Completion': ['8.5h', '6.2h', '4.1h', '9.8h', '3.5h']
        })
        
        st.dataframe(
            cluster_data.style.background_gradient(subset=['Customers', 'Avg Priority Score']),
            use_container_width=True
        )
    
    with tab2:
        st.subheader("🚛 AI-Driven Crew Routing & Assignment")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Crew Status Dashboard")
            
            # Crew status counts
            status_counts = optimizer.crews['status'].value_counts()
            
            # Create crew status chart
            fig_status = go.Figure(data=[
                go.Bar(
                    x=status_counts.index,
                    y=status_counts.values,
                    marker_color=['#2ed573', '#ff6b6b', '#ffa502'],
                    text=status_counts.values,
                    textposition='auto'
                )
            ])
            fig_status.update_layout(
                title="Crew Deployment Status",
                xaxis_title="Status",
                yaxis_title="Number of Crews",
                height=300
            )
            st.plotly_chart(fig_status, use_container_width=True)
            
            # Optimization metrics
            st.markdown("#### 🎯 Optimization Performance")
            metrics_df = pd.DataFrame({
                'Metric': [
                    'Resource Utilization',
                    'Travel Time Reduction',
                    'Priority Completion Rate',
                    'Overall Optimization Score'
                ],
                'Value': [
                    f"{optimizer.optimization_results['resource_utilization']:.1f}%",
                    optimizer.optimization_results['travel_time_reduction'],
                    f"{optimizer.optimization_results['priority_completion_rate']:.1f}%",
                    f"{optimizer.optimization_results['optimization_score']:.1f}/100"
                ]
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### Optimal Crew Assignments")
            
            # Generate optimal assignments
            assignments = []
            available_crews = optimizer.crews[optimizer.crews['status'] == 'Available']
            high_priority_outages = optimizer.outages[optimizer.outages['priority'] == 'High'].head(len(available_crews))
            
            for i, (crew_idx, crew) in enumerate(available_crews.iterrows()):
                if i < len(high_priority_outages):
                    outage = high_priority_outages.iloc[i]
                    assignments.append({
                        'Crew': crew['crew_id'],
                        'Assignment': outage['outage_id'],
                        'Priority': outage['priority'],
                        'Customers': outage['customers_affected'],
                        'Travel Time': f"{np.random.uniform(0.3, 1.2):.1f}h",
                        'Est. Completion': f"{np.random.uniform(2, 8):.1f}h"
                    })
            
            if assignments:
                assignments_df = pd.DataFrame(assignments)
                st.dataframe(
                    assignments_df.style.applymap(
                        lambda x: 'color: #ff4757; font-weight: bold' if x == 'High' else '',
                        subset=['Priority']
                    ),
                    use_container_width=True,
                    hide_index=True
                )
            
            # Travel optimization visualization
            st.markdown("#### 🗺️ Route Optimization")
            
            # Create route efficiency chart
            hours = list(range(24))
            baseline_time = [np.random.uniform(2, 4) for _ in hours]
            optimized_time = [time * 0.55 for time in baseline_time]  # 45% reduction
            
            fig_routes = go.Figure()
            fig_routes.add_trace(go.Scatter(
                x=hours, y=baseline_time, name='Baseline Routing',
                line=dict(color='#ff6b6b', width=2)
            ))
            fig_routes.add_trace(go.Scatter(
                x=hours, y=optimized_time, name='AI-Optimized Routing',
                line=dict(color='#2ed573', width=2)
            ))
            fig_routes.update_layout(
                title="Average Travel Time by Hour",
                xaxis_title="Hour of Day",
                yaxis_title="Travel Time (hours)",
                height=300
            )
            st.plotly_chart(fig_routes, use_container_width=True)
    
    with tab3:
        st.subheader("📦 Material & Resource Planning")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Inventory Status")
            
            # Materials inventory chart
            materials_chart = optimizer.materials.copy()
            materials_chart['Usage %'] = (materials_chart['required'] / materials_chart['available'] * 100).round(1)
            
            fig_materials = px.bar(
                materials_chart,
                x='material',
                y=['available', 'required'],
                title="Material Availability vs Requirements",
                barmode='group',
                color_discrete_map={'available': '#2ed573', 'required': '#ff6b6b'}
            )
            fig_materials.update_layout(height=400)
            fig_materials.update_xaxis(tickangle=45)
            st.plotly_chart(fig_materials, use_container_width=True)
            
        with col2:
            st.markdown("#### Critical Shortages Alert")
            
            # Identify potential shortages
            materials_status = optimizer.materials.copy()
            materials_status['Utilization'] = materials_status['required'] / materials_status['available']
            materials_status['Status'] = materials_status['Utilization'].apply(
                lambda x: 'Critical' if x > 0.8 else 'Warning' if x > 0.6 else 'Good'
            )
            
            # Display status table
            status_display = materials_status[['material', 'available', 'required', 'Status']].copy()
            
            def color_status(val):
                if val == 'Critical':
                    return 'background-color: #ff4757; color: white; font-weight: bold'
                elif val == 'Warning':
                    return 'background-color: #ffa502; color: white; font-weight: bold'
                else:
                    return 'background-color: #2ed573; color: white; font-weight: bold'
            
            st.dataframe(
                status_display.style.applymap(color_status, subset=['Status']),
                use_container_width=True,
                hide_index=True
            )
            
            # Warehouse locations
            st.markdown("#### 🏪 Warehouse Locations")
            warehouse_data = pd.DataFrame({
                'Warehouse': ['Warehouse A', 'Warehouse B', 'Warehouse C'],
                'Distance (miles)': [12.5, 8.3, 15.7],
                'Capacity': ['High', 'Medium', 'High'],
                'Accessibility': ['Good', 'Excellent', 'Moderate']
            })
            st.dataframe(warehouse_data, use_container_width=True, hide_index=True)
        
        # Material demand forecasting
        st.markdown("#### 📈 Real-time Demand Forecasting")
        
        # Create demand forecast chart
        time_periods = [f"Hour {i}" for i in range(1, 13)]
        
        materials_demand = {
            'Poles': np.random.poisson(3, 12),
            'Wire': np.random.poisson(8, 12),
            'Transformers': np.random.poisson(2, 12),
            'Switches': np.random.poisson(1, 12)
        }
        
        fig_demand = go.Figure()
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
        
        for i, (material, demand) in enumerate(materials_demand.items()):
            fig_demand.add_trace(go.Scatter(
                x=time_periods,
                y=demand,
                name=material,
                line=dict(color=colors[i], width=3),
                mode='lines+markers'
            ))
        
        fig_demand.update_layout(
            title="Predicted Material Demand - Next 12 Hours",
            xaxis_title="Time Period",
            yaxis_title="Units Required",
            height=400
        )
        st.plotly_chart(fig_demand, use_container_width=True)
    
    with tab4:
        st.subheader("📊 Real-time Performance Analytics")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### ⚡ Restoration Progress")
            
            # Restoration timeline
            hours_elapsed = list(range(0, 13))
            outages_remaining = [45, 42, 38, 35, 30, 26, 22, 18, 15, 12, 8, 5, 2]
            customers_restored = [0, 1200, 3500, 6800, 12000, 18500, 25000, 32000, 38500, 44000, 48500, 52000, 54500]
            
            fig_progress = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_progress.add_trace(
                go.Scatter(x=hours_elapsed, y=outages_remaining, name="Outages Remaining",
                          line=dict(color='#ff6b6b', width=3)),
                secondary_y=False,
            )
            
            fig_progress.add_trace(
                go.Scatter(x=hours_elapsed, y=customers_restored, name="Customers Restored",
                          line=dict(color='#2ed573', width=3)),
                secondary_y=True,
            )
            
            fig_progress.update_xaxis(title_text="Hours Since Storm")
            fig_progress.update_yaxis(title_text="Outages Remaining", secondary_y=False)
            fig_progress.update_yaxis(title_text="Customers Restored", secondary_y=True)
            fig_progress.update_layout(height=400, title="Restoration Progress Timeline")
            
            st.plotly_chart(fig_progress, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Performance Metrics")
            
            # Key performance indicators
            kpi_data = pd.DataFrame({
                'KPI': [
                    'Customer Minutes Interrupted',
                    'Average Restoration Time',
                    'Crew Utilization Rate',
                    'First-Time Fix Rate',
                    'Emergency Response Time',
                    'Customer Satisfaction'
                ],
                'Current': [
                    '2.4M',
                    '4.2 hrs',
                    '94.2%',
                    '87.5%',
                    '18 min',
                    '4.3/5.0'
                ],
                'Target': [
                    '< 3.0M',
                    '< 5.0 hrs',
                    '> 90%',
                    '> 85%',
                    '< 20 min',
                    '> 4.0/5.0'
                ],
                'Status': [
                    '✅ On Track',
                    '✅ Ahead',
                    '✅ Excellent',
                    '✅ Good',
                    '✅ Excellent',
                    '✅ Good'
                ]
            })
            
            st.dataframe(kpi_data, use_container_width=True, hide_index=True)
            
            # Cost savings analysis
            st.markdown("#### 💰 Cost Impact Analysis")
            
            cost_data = pd.DataFrame({
                'Category': ['Labor Costs', 'Equipment Rental', 'Material Costs', 'Customer Credits'],
                'Baseline ($K)': [850, 320, 180, 450],
                'Optimized ($K)': [720, 280, 165, 320],
                'Savings ($K)': [130, 40, 15, 130]
            })
            
            fig_cost = px.bar(
                cost_data,
                x='Category',
                y=['Baseline ($K)', 'Optimized ($K)'],
                title="Cost Optimization Results",
                barmode='group',
                color_discrete_map={'Baseline ($K)': '#ff6b6b', 'Optimized ($K)': '#2ed573'}
            )
            fig_cost.update_layout(height=300)
            st.plotly_chart(fig_cost, use_container_width=True)
        
        # Real-time decision support
        st.markdown("#### 🤖 AI Decision Support Recommendations")
        
        recommendations = [
            "🎯 **Priority Adjustment**: Redirect CREW_08 from OUT_025 to OUT_017 (higher customer impact)",
            "📦 **Resource Alert**: Request additional transformers from Warehouse B (shortage predicted in 3 hours)",
            "🚛 **Route Optimization**: Alternative route for CREW_03 via I-95 (traffic detected on current path)",
            "⚡ **Load Balancing**: Consider temporary switching to reduce load on Feeder F-12 during restoration",
            "📱 **Communication**: Update ETA for high-priority customers (delays detected in North Miami cluster)"
        ]
        
        for rec in recommendations:
            st.markdown(f"• {rec}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <h3>⚡ Storm Restoration Optimization Platform</h3>
        <p>Built for grid decarbonization and sustainable energy integration</p>
        <p><strong>🔄 Prophet + XGBoost Forecasting</strong> • <strong>🧠 K-Means Clustering</strong> • <strong>📱 Pyomo Optimization</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
