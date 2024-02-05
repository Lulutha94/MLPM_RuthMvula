import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Initialize random seed for reproducibility
np.random.seed(42)

# Generate date range for 10 years
start_date = datetime.now() - timedelta(days=365 * 20)
end_date = datetime.now()
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Define component IDs and cost ranges
num_components = 100
component_ids = np.arange(1, num_components + 1)
component_cost_ranges = {
    'Rail': {
        'Preventive': (1000, 3000),
        'Corrective': (5000, 20000),
    },
    'Switch': {
        'Preventive': (2500, 7000),
        'Corrective': (7000, 20000),
    },
    'Signal': {
        'Preventive': (1200, 3500),
        'Corrective': (5000, 18000),
    },
    'Track Bed': {
        'Preventive': (1500, 4000),
        'Corrective': (6000, 25000),
    },
    'Gravel': {
        'Preventive': (50, 100),
        'Corrective': (2500, 5000),
    },

}

# Initialize data structures
data = {}
num_records = 100000
maintenance_history = {component_id: datetime.min for component_id in component_ids}
preventive_counter = {component: 0 for component in component_cost_ranges}

# Define lists for storing maintenance data
components_maintained = []
components_ids_maintained = []
maintenance_costs = []
maintenance_durations = []
maintenance_types = []
maintenance_dates = []
weather_temperature = []
weather_humidity = []
weather_rainfall = []
weather_windspeed = []
weather_solar = []
weather_snow = []
traffic_load_train_size = []
traffic_load_coach_number = []
traffic_load_acceleration = []
traffic_load_speed = []
traffic_load_train_type = []
traffic_load_total_weight = []
track_geometry_length = []
track_geometry_width = []
track_geometry_curvature = []
track_geometry_gradient = []

# Function to generate weather data based on date
def generate_weather_data(date):
    month = date.month

    if 12 <= month or month <= 2:  # Winter: December to February
        temperature = np.random.randint(-10, 5)  # Winter temperatures
        snow = 'Yes' if np.random.rand() > 0.5 else 'No'
        humidity = np.random.randint(70, 90)
        rainfall = np.random.uniform(10, 60)  # More rainfall/snowfall
        wind_speed = np.random.uniform(10, 40)
        solar_radiation = np.random.uniform(50, 200)  # Lower solar radiation
    elif 3 <= month <= 5:  # Spring: March to May
        temperature = np.random.randint(0, 15)  # Spring temperatures
        snow = 'No'
        humidity = np.random.randint(60, 80)
        rainfall = np.random.uniform(30, 70)
        wind_speed = np.random.uniform(10, 30)
        solar_radiation = np.random.uniform(200, 500)
    elif 6 <= month <= 8:  # Summer: June to August
        temperature = np.random.randint(15, 30)  # Summer temperatures
        snow = 'No'
        humidity = np.random.randint(50, 70)
        rainfall = np.random.uniform(20, 80)  # Occasional heavy rainfall
        wind_speed = np.random.uniform(5, 25)
        solar_radiation = np.random.uniform(400, 800)  # Higher solar radiation
    else:  # Fall: September to November
        temperature = np.random.randint(5, 15)  # Fall temperatures
        snow = 'No'
        humidity = np.random.randint(65, 85)
        rainfall = np.random.uniform(30, 80)
        wind_speed = np.random.uniform(10, 35)
        solar_radiation = np.random.uniform(150, 450)

    return temperature, humidity, rainfall, wind_speed, solar_radiation, snow

# Iterate and generate maintenance records
for _ in range(num_records):
    date = np.random.choice(date_range)
    component = np.random.choice(['Rail', 'Switch', 'Signal', 'Track Bed', 'Gravel'])
    component_id = np.random.choice(component_ids)
    maintenance_type = np.random.choice(['Preventive', 'Corrective'])
    
    last_maintenance = maintenance_history[component_id]

    if isinstance(date, np.datetime64):
        date = date.astype('M8[ms]').astype(datetime)
    
    years_since_last = (date - last_maintenance).days / 365

    if years_since_last >= 3:
        maintenance_type = 'Preventive'
        preventive_counter[component] += 1
    else:
        maintenance_type = np.random.choice(['Preventive', 'Corrective'])

    cost_range = component_cost_ranges[component][maintenance_type]
    cost = np.random.randint(cost_range[0], cost_range[1] + 1)

    # duration_range_multiplier = 0.05 if maintenance_type == 'Preventive' else 0.5
    # max_duration = int(cost_range[1] * duration_range_multiplier)  # Maximum duration based on cost

    if maintenance_type == 'Preventive':
        max_duration = 3  # Preventive maintenance takes up to 3 days
    else:  # Corrective
        max_duration = 7  # Corrective maintenance takes up to 7 days


    duration = np.random.randint(1, max_duration + 1)  # Duration in days
    
    formatted_date = date.strftime('%Y-%m-%d')
    
    components_maintained.append(component)
    components_ids_maintained.append(component_id)
    maintenance_costs.append(cost)
    maintenance_durations.append(duration)
    maintenance_types.append(maintenance_type)
    maintenance_dates.append(formatted_date)
    maintenance_history[component_id] = date

    # Update weather data
    weather_data = generate_weather_data(date)
    
    # ... (append weather data to corresponding lists)        
    weather_temperature.append(weather_data[0])
    weather_humidity.append(weather_data[1])
    weather_rainfall.append(weather_data[2])
    weather_windspeed.append(weather_data[3])
    weather_solar.append(weather_data[4])
    weather_snow.append(weather_data[5])
    # ... (append weather data to corresponding lists)
            
    traffic_load_train_size.append(np.random.choice(['Small', 'Medium', 'Large']))
    traffic_load_coach_number.append(np.random.randint(5, 15))
    traffic_load_acceleration.append(np.random.uniform(0.5, 2.3))
    traffic_load_speed.append(np.random.randint(50, 230))
    traffic_load_train_type.append(np.random.choice(['Light Rail', 'Cargo Train', 'Passenger Train']))
    traffic_load_total_weight.append(np.random.randint(500, 2000))  # in tons
    track_geometry_length.append(np.random.randint(500, 2000))  # in kilometers
    track_geometry_width.append(np.random.uniform(1.4, 1.7))  # in meters
    track_geometry_curvature.append(np.random.uniform(0.1, 0.5))
    track_geometry_gradient.append(np.random.uniform(0.01, 0.1))


# Ensure preventive maintenance for each component every three years
current_date = datetime.now()
for component, count in preventive_counter.items():
    while count < 1:
        component_id = np.random.choice(component_ids)
        last_maintenance = maintenance_history[component_id]

        if isinstance(current_date, np.datetime64):
            current_date = date.astype('M8[ms]').astype(datetime)

        years_since_last = (current_date - last_maintenance).days / 365

        # Generate preventive maintenance date within the last three years
        if years_since_last >= 3:
            maintenance_date = last_maintenance + timedelta(days=365 * np.random.uniform(3, 5))
            if maintenance_date > current_date:
                maintenance_date = current_date

            # Generate maintenance data
            cost_range = component_cost_ranges[component]['Preventive']
            cost = np.random.randint(cost_range[0], cost_range[1] + 1)

            # duration_range_multiplier = 0.05
            # max_duration = int(cost_range[1] * duration_range_multiplier)  # Maximum duration based on cost
            if maintenance_type == 'Preventive':
                max_duration = 3  # Preventive maintenance takes up to 3 days
            else:  # Corrective
                max_duration = 7  # Corrective maintenance takes up to 7 days


            duration = np.random.randint(1, max_duration + 1)  # Duration in days
            
            formatted_date = maintenance_date.strftime('%Y-%m-%d')

            # Append the data
            maintenance_costs.append(cost)
            maintenance_durations.append(duration)
            maintenance_types.append('Preventive')
            maintenance_dates.append(formatted_date)
            maintenance_history[component_id] = maintenance_date

            # Update weather data
            weather_data = generate_weather_data(maintenance_date)
            # ... (append weather data to corresponding lists)
                    
            weather_temperature.append(weather_data[0])
            weather_humidity.append(weather_data[1])
            weather_rainfall.append(weather_data[2])
            weather_windspeed.append(weather_data[3])
            weather_solar.append(weather_data[4])
            weather_snow.append(weather_data[5])

            count += 1




# Convert lists to DataFrame
df = pd.DataFrame({
    'Date': maintenance_dates,
    'Component': components_maintained,
    'Component_ID': components_ids_maintained,
    'Maintenance_Cost': maintenance_costs,
    'Maintenance_Duration': maintenance_durations,
    'Maintenance_Type': maintenance_types,
    'Traffic_Load_Train_Size': traffic_load_train_size,  # Randomly generated in your loop
    'Traffic_Load_Coach_Number': traffic_load_coach_number,  # Randomly generated
    'Traffic_Load_Acceleration': traffic_load_acceleration,  # Randomly generated
    'Traffic_Load_Speed': traffic_load_speed,  # Randomly generated
    'Traffic_Load_Train_Type': traffic_load_train_type,  # Randomly generated
    'Traffic_Load_Total_Weight': traffic_load_total_weight,  # Randomly generated
    'Track_Geometry_Length': track_geometry_length,  # Randomly generated
    'Track_Geometry_Width': track_geometry_width,  # Randomly generated
    'Track_Geometry_Curvature': track_geometry_curvature,  # Randomly generated
    'Track_Geometry_Gradient': track_geometry_gradient,  # Randomly generated
    'Weather_Air_Temperature': weather_temperature,  # Generated based on date
    'Weather_Relative_Humidity': weather_humidity,  # Randomly generated
    'Weather_Rainfall': weather_rainfall,  # Randomly generated
    'Weather_Wind_Speed': weather_windspeed,  # Randomly generated
    'Weather_Solar_Radiation': weather_solar,  # Randomly generated
    'Weather_Snow': weather_snow  # Generated based on date
})

df.to_csv('simulated_maintenance_data.csv', index=False)