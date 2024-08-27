import streamlit as st
import pandas as pd
import pulp
from datetime import datetime, timedelta
import numpy as np

# Initialize session state
if 'tabs' not in st.session_state:
    st.session_state.tabs = []  # Store tab data as a list of dictionaries
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = None
if 'scenario_exclusions' not in st.session_state:
    st.session_state.scenario_exclusions = {}  # Track exclusions per scenario
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame()  # Initialize empty DataFrame
if 'data_retrieved' not in st.session_state:
    st.session_state.data_retrieved = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()  # Store results data
if 'last_fetched' not in st.session_state:
    st.session_state.last_fetched = None  # Track last fetch time
if 'selected_trip_details' not in st.session_state:
    st.session_state.selected_trip_details = pd.DataFrame()  # Initialize empty DataFrame for details

def generate_sample_data(num_trips):
    # Generate sample trip data
    start_date = datetime.now()
    sample_data = []
    
    for i in range(num_trips):
        trip_id = i + 1
        trip_start_date = start_date + timedelta(days=i)
        trip_end_date = trip_start_date + timedelta(days=np.random.randint(1, 3))
        distance = np.random.randint(50, 300)
        duration = np.random.randint(1, 10)
        savings = np.random.randint(100, 500)
        
        sample_data.append({
            'Trip ID': trip_id,
            'Start Date': trip_start_date.strftime('%Y-%m-%d'),
            'End Date': trip_end_date.strftime('%Y-%m-%d'),
            'Distance': distance,
            'Duration': duration,
            'Savings': savings
        })
    
    return pd.DataFrame(sample_data)

def generate_details_data(trip_id):
    # Generate random details data for a specific trip ID
    num_details = 10
    return pd.DataFrame([{
        'Trip ID': trip_id,
        'Detail 1': np.random.random(),
        'Detail 2': np.random.random(),
        'Detail 3': np.random.random()
    } for _ in range(num_details)])


def run_optimization(min_savings, max_distance, max_duration, excluded_trip_ids):
    filtered_data = [trip for trip in st.session_state.data.to_dict('records') if trip['Trip ID'] not in excluded_trip_ids]
    filtered_data = [trip for trip in filtered_data if trip['Savings'] >= min_savings and trip['Distance'] <= max_distance and trip['Duration'] <= max_duration]

    model = pulp.LpProblem("TripOptimization", pulp.LpMaximize)
    trips = [trip['Trip ID'] for trip in filtered_data]
    x = pulp.LpVariable.dicts("Trip", trips, cat=pulp.LpBinary)

    model += pulp.lpSum([trip['Savings'] * x[trip['Trip ID']] for trip in filtered_data]) - \
             pulp.lpSum([trip['Distance'] * x[trip['Trip ID']] for trip in filtered_data]) - \
             pulp.lpSum([trip['Duration'] * x[trip['Trip ID']] for trip in filtered_data])

    model += pulp.lpSum([x[trip['Trip ID']] for trip in filtered_data]) <= 2

    model.solve()

    results = {
        'Trip ID': [],
        'Selected': [],
        'Savings': [],
        'Distance': [],
        'Duration': []
    }

    for trip in filtered_data:
        results['Trip ID'].append(trip['Trip ID'])
        results['Selected'].append(x[trip['Trip ID']].value())
        results['Savings'].append(trip['Savings'])
        results['Distance'].append(trip['Distance'])
        results['Duration'].append(trip['Duration'])

    results_df = pd.DataFrame(results)

    excluded_data = [trip for trip in st.session_state.data.to_dict('records') if trip['Trip ID'] in excluded_trip_ids]
    excluded_df = pd.DataFrame(excluded_data)

    combined_df = pd.concat([results_df, excluded_df.rename(columns=lambda x: f"Excluded_{x}")], axis=1)

    return results_df, excluded_df, combined_df, pulp.LpStatus[model.status], pulp.value(model.objective)

# Trip History Section
st.sidebar.header('Step 1: Trip History')

start_date = st.sidebar.date_input('Start Date', value=datetime(2023, 11, 1).date(), format='MM/DD/YYYY')
end_date = st.sidebar.date_input('End Date', value=datetime(2023, 11, 6).date(), format='MM/DD/YYYY')

# Button for retrieving trip history information
if st.sidebar.button('Get Trip History'):
    try:
        st.session_state.data = generate_sample_data(200)
        st.session_state.data_retrieved = True
        st.session_state.last_fetched = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.sidebar.success('Data updated successfully')
    except Exception as e:
        st.sidebar.error(f"Error retrieving data: {e}")
        st.session_state.data_retrieved = False

# Add horizontal line
st.sidebar.markdown('---')

# Find Matching Trip(s) Section
st.sidebar.header('Step 2: Find Matching Trip(s)')

min_savings = st.sidebar.number_input('Minimum Savings', min_value=0, value=0, key='min_savings')
max_distance = st.sidebar.number_input('Maximum Distance', min_value=0, value=1000, key='max_distance')
max_duration = st.sidebar.number_input('Maximum Duration', min_value=0, value=1000, key='max_duration')

# Radio button to select tab
tab_titles = [tab['title'] for tab in st.session_state.tabs] if st.session_state.tabs else ['No Scenario']
selected_tab = st.sidebar.radio("Select Scenario", options=tab_titles)

# Update selected tab in session state
st.session_state.selected_tab = selected_tab

# Determine available exclusions based on the selected tab
if selected_tab and selected_tab != 'No Scenario':
    selected_tab_data = next((tab for tab in st.session_state.tabs if tab['title'] == selected_tab), None)
    if selected_tab_data:
        available_trips = [trip['Trip ID'] for trip in st.session_state.data.to_dict('records')]
        current_scenario_exclusions = [trip['Trip ID'] for trip in selected_tab_data['excluded_data'].to_dict('records')]
    else:
        available_trips = [trip['Trip ID'] for trip in st.session_state.data.to_dict('records')]
        current_scenario_exclusions = []
else:
    available_trips = [trip['Trip ID'] for trip in st.session_state.data.to_dict('records')]
    current_scenario_exclusions = []

# Multiselect widget for trip exclusions
selected_excluded_trip_ids = st.sidebar.multiselect(
    'Select trips to exclude',
    options=available_trips,
    default=current_scenario_exclusions,
    key='exclude_trips'
)

# Update the session state with the selected exclusions for the current scenario
if selected_tab:
    st.session_state.scenario_exclusions[selected_tab] = list(set(selected_excluded_trip_ids))

# Run Scenario button (disabled if data has not been retrieved)
run_scenario_button_disabled = not st.session_state.get('data_retrieved', False)
if st.sidebar.button('Run Scenario', key='run_scenario', disabled=run_scenario_button_disabled):
    if selected_tab and selected_tab != 'No Scenario':
        # Fetch data from the selected tab and include exclusions
        selected_tab_data = next((tab for tab in st.session_state.tabs if tab['title'] == selected_tab), None)
        if selected_tab_data:
            previous_exclusions = [trip['Trip ID'] for trip in selected_tab_data['excluded_data'].to_dict('records')]
            combined_exclusions = list(set(previous_exclusions + selected_excluded_trip_ids))
            filtered_data = [trip for trip in selected_tab_data['filtered_data'] if trip['Trip ID'] not in combined_exclusions]
            filtered_data = [trip for trip in filtered_data if trip['Savings'] >= min_savings and trip['Distance'] <= max_distance and trip['Duration'] <= max_duration]
        else:
            filtered_data = st.session_state.data.to_dict('records')
    else:
        filtered_data = st.session_state.data.to_dict('records')

    # Apply filter criteria
    filtered_data = [trip for trip in filtered_data if trip['Savings'] >= min_savings and trip['Distance'] <= max_distance and trip['Duration'] <= max_duration]
    
    results_df, excluded_df, combined_df, status, total_savings = run_optimization(
        min_savings,
        max_distance,
        max_duration,
        st.session_state.scenario_exclusions.get(selected_tab, [])
    )

    # Add results to tabs at the beginning of the list
    tab_title = f"Scenario {len(st.session_state.tabs) + 1}"
    st.session_state.tabs.insert(0, {
        'title': tab_title,
        'filtered_data': filtered_data,
        'excluded_data': pd.DataFrame([trip for trip in st.session_state.data.to_dict('records') if trip['Trip ID'] in st.session_state.scenario_exclusions.get(selected_tab, [])]),
        'combined_df': combined_df,
        'status': status,
        'total_savings': total_savings
    })

    # Set the newly created tab as the selected tab
    st.session_state.selected_tab = tab_title

    # Force a rerun to ensure the new tab is selected
    st.rerun()

# Layout for displaying results in rows
st.markdown("<h1 style='text-align: center; background-color: darkgreen; color: white; padding: 10px;'>Trip Optimization Dashboard</h1>", unsafe_allow_html=True)

# Top row: Trip History Information
st.subheader("Trip History Information")
if st.session_state.data_retrieved:
    st.dataframe(st.session_state.data, width=450, height=300)
    if st.session_state.last_fetched:
        st.write(f"Last fetched: {st.session_state.last_fetched}")
else:
    st.write("No trip history data available.")

# Divider
st.markdown('---')

# Bottom row: Scenario, Header, and Details Grids
if st.session_state.selected_tab:
    st.subheader("Selected Scenario and Excluded Trips")
    if st.session_state.selected_tab:
        st.write(f"Scenario: {st.session_state.selected_tab}")

        if st.session_state.selected_tab != 'No Scenario':
            selected_tab_data = next((tab for tab in st.session_state.tabs if tab['title'] == st.session_state.selected_tab), None)
            if selected_tab_data:
                excluded_trips = [trip['Trip ID'] for trip in selected_tab_data['excluded_data'].to_dict('records')]
                st.write("Excluded Trips:")
                st.write(excluded_trips)

    # Create two rows for Header and Details grids
    st.markdown('<hr>', unsafe_allow_html=True)

    # Header grid
    st.subheader("Header")
    if st.session_state.data_retrieved:
        if st.session_state.selected_tab:
            selected_tab_data = next((tab for tab in st.session_state.tabs if tab['title'] == st.session_state.selected_tab), None)
            if selected_tab_data:
                header_df = pd.DataFrame(selected_tab_data['filtered_data'])
                selected_trip_id = st.selectbox("Select a Trip ID", header_df['Trip ID'], key='trip_selector')
                st.session_state.selected_trip_details = generate_details_data(selected_trip_id)
                st.dataframe(header_df, width=450, height=300)

    # Details grid
    st.subheader("Details")
    if not st.session_state.selected_trip_details.empty:
        st.dataframe(st.session_state.selected_trip_details, width=450, height=300)
    else:
        st.write("Select a trip from the Header grid to see details here.")
