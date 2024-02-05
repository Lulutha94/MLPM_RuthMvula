# -*- coding: utf-8 -*-
"""
Created on Tuesday Jan 30 17:27:07 2024

@author: Ruth Mvula

"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Initialize random seed for reproducibility
np.random.seed(42)

# Generate date range for 20 years
start_date = datetime.now() - timedelta(days=365 * 20)
end_date = datetime.now()
date_range = pd.date_range(start=start_date, end=end_date, freq="W")

# Define component IDs and cost ranges
num_components = 100
component_ids = np.arange(1, num_components + 1)
component_cost_ranges = {
    "Rail": {
        "Preventive": (1000, 3000),
        "Corrective": (5000, 20000),
    },
    "Switch": {
        "Preventive": (2500, 7000),
        "Corrective": (7000, 20000),
    },
    "Signal": {
        "Preventive": (1200, 3500),
        "Corrective": (5000, 18000),
    },
    "Track Bed": {
        "Preventive": (1500, 4000),
        "Corrective": (6000, 25000),
    },
    "Gravel": {
        "Preventive": (50, 100),
        "Corrective": (2500, 5000),
    },
}

# installation_dates = {(component_id, np.random.choice(list(component_cost_ranges.keys()))): start_date for component_id in component_ids}

component_id_to_type = {
    component_id: np.random.choice(list(component_cost_ranges.keys()))
    for component_id in component_ids
}

installation_dates = {
    (component_id, component_id_to_type[component_id]): start_date
    - timedelta(days=np.random.randint(0, 365 * 20))
    for component_id in range(1, num_components + 1)
}

last_failure_dates = {key: start_date for key in installation_dates}


# Initialize data structures
data = {}
num_records = 100000
maintenance_history = {component_id: datetime.min for component_id in component_ids}
preventive_counter = {component: 0 for component in component_cost_ranges}

# Define lists for storing maintenance data
components_loop = []
components_ids_loop = []

maintenance_costs = []
maintenance_durations = []
maintenance_types = []
maintenance_dates = []

dates = []
component_ids_list = []
component_types_list = []
component_ages = []
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
time_since_last_maintenance = []
number_of_maintenances = []
recent_maintenance_type = []
failures = []


# Function to generate weather data based on date
def generate_weather_data(date):
    month = date.month

    if 12 <= month or month <= 2:  # Winter: December to February
        temperature = np.random.randint(-10, 5)  # Winter temperatures
        snow = "Yes" if np.random.rand() > 0.5 else "No"
        humidity = np.random.randint(70, 90)
        rainfall = np.random.uniform(10, 60)  # More rainfall/snowfall
        wind_speed = np.random.uniform(10, 40)
        solar_radiation = np.random.uniform(50, 200)  # Lower solar radiation
    elif 3 <= month <= 5:  # Spring: March to May
        temperature = np.random.randint(0, 15)  # Spring temperatures
        snow = "No"
        humidity = np.random.randint(60, 80)
        rainfall = np.random.uniform(30, 70)
        wind_speed = np.random.uniform(10, 30)
        solar_radiation = np.random.uniform(200, 500)
    elif 6 <= month <= 8:  # Summer: June to August
        temperature = np.random.randint(15, 30)  # Summer temperatures
        snow = "No"
        humidity = np.random.randint(50, 70)
        rainfall = np.random.uniform(20, 80)  # Occasional heavy rainfall
        wind_speed = np.random.uniform(5, 25)
        solar_radiation = np.random.uniform(400, 800)  # Higher solar radiation
    else:  # Fall: September to November
        temperature = np.random.randint(5, 15)  # Fall temperatures
        snow = "No"
        humidity = np.random.randint(65, 85)
        rainfall = np.random.uniform(30, 80)
        wind_speed = np.random.uniform(10, 35)
        solar_radiation = np.random.uniform(150, 450)

    weather_data = {
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "wind_speed": wind_speed,
        "solar_radiation": solar_radiation,
        "snow": snow,
    }
    return weather_data


def generate_maintenance_data(component_id, date):
    # Simplified logic to generate maintenance data
    # This function should be more sophisticated in a real scenario
    last_maintenance = datetime.now() - timedelta(days=np.random.randint(0, 365 * 3))
    num_maintenances = np.random.randint(0, 10)
    recent_maintenance = np.random.choice(["Preventive", "Corrective"])
    return (date - last_maintenance).days, num_maintenances, recent_maintenance


def generate_traffic_data():
    traffic_data = {
        "train_size": np.random.choice(["Small", "Medium", "Large"]),
        "coach_number": np.random.randint(5, 15),
        "acceleration": np.random.uniform(0.5, 2.3),
        "speed": np.random.randint(50, 230),
        "train_type": np.random.choice(
            ["Light Rail", "Cargo Train", "Passenger Train"]
        ),
        "total_weight": np.random.randint(500, 2000),  # in tons
    }
    return traffic_data


def generate_track_geometry():
    track_data = {
        "length": np.random.randint(500, 2000),  # Length in kilometers
        "width": np.random.uniform(1.4, 1.7),  # Width in meters
        "curvature": np.random.uniform(0.1, 0.5),
        "gradient": np.random.uniform(0.01, 0.1),
    }
    return track_data


def should_fail(
    component_type,
    component_age,
    time_since_maintenance,
    weather_data,
    traffic_data,
    track_geometry,
):
    # Base failure probability
    base_probability = 0.03  # Starting with a 3% base chance of failure

    # Adjust base probability based on component type
    type_factor = {
        "Rail": 1.1,
        "Switch": 1.3,
        "Signal": 1.2,
        "Track Bed": 1.05,
        "Gravel": 0.0,
    }

    base_probability *= type_factor.get(component_type, 1)

    # Adjust age factor to be more gradual
    age_factor = (
        1 + (component_age / 365) ** 1.1
    )  # Use power of 1.5 instead of 2 for a slower increase

    failure_probability = base_probability * age_factor

    # Increase probability based on time since last maintenance
    if time_since_maintenance > 365:  # More than a year
        failure_probability *= 1.25  # Increase by 25%
    elif time_since_maintenance > 730:  # More than two years
        failure_probability *= 1.35  # Increase by 35%

    # Adjust probability based on extreme weather conditions
    if weather_data["temperature"] < -10 or weather_data["temperature"] > 35:
        failure_probability *= 1.2  # Increase by 2% for extreme temperatures
    if weather_data["snow"] == "Yes":
        failure_probability *= 1.2  # Increase by 2% for snowy conditions

    # Consider operational factors
    if (
        traffic_data["train_size"] == "Large"
        or traffic_data["speed"] > 180
        or traffic_data["total_weight"] > 1500
    ):
        failure_probability *= 1.3  # Increase by 3% for high load or high speed

    # Account for track geometry
    if track_geometry["curvature"] > 0.3 or track_geometry["gradient"] > 0.07:
        failure_probability *= (
            1.3  # Increase by 3% for sharp curvature or steep gradient
        )

    # Check if the component is 'Gravel' or less than 6 months old and exclude it from failing
    if component_type == "Gravel" or component_age < 180:
        return False

    # Apply an additional age factor that starts later and increases more slowly
    if component_age > 365:  # Start increasing more after one year
        # Grow more slowly with a smaller exponent
        additional_age_factor = np.log1p((component_age - 365) / 365)
        failure_probability += additional_age_factor

    # Ensure the probability does not exceed 50%
    failure_probability = min(failure_probability, 0.5)

    # Random chance of failure based on calculated probability
    return np.random.rand() < failure_probability


# Iterate and generate data
replacement_period = 365 * 5
minimum_age_for_replacement = 365 * 5

for date in date_range:
    for component_id in component_ids:
        component_type = component_id_to_type[component_id]

        components_loop.append(component_type)
        components_ids_loop.append(component_id)

        key = (component_id, component_type)

        if isinstance(date, np.datetime64):
            date = date.astype("M8[ms]").astype(datetime)

        # print(date, installation_dates[key])
        time_since_last = (date - installation_dates[key]).days
        time_since_last_failure = (date - last_failure_dates[key]).days

        if time_since_last >= replacement_period or (
            failures and time_since_last_failure >= replacement_period
        ):
            installation_dates[key] = date  # Simulate component replacement or repair
            last_failure_dates[key] = date  # Reset last failure date
            time_since_last = 0
        # age = time_since_last / 365  # Approximate component age

        formatted_date = date.strftime("%Y-%m-%d")

        # Generate environmental and operational data
        weather_data = generate_weather_data(date)
        maintenance_data = generate_maintenance_data(component_id, date)
        traffic_load = generate_traffic_data()
        track_geometry = generate_track_geometry()

        failure_occurred = should_fail(
            component_type,
            time_since_last,
            time_since_last,
            weather_data,
            traffic_load,
            track_geometry,
        )

        # If a failure has occurred and the component is older than the minimum age for replacement
        if failure_occurred and time_since_last >= minimum_age_for_replacement:
            installation_dates[key] = date  # Replace the component with a new one
            last_failure_dates[key] = date  # Reset the last failure date
        elif failure_occurred:
            # If the component has failed but is not old enough to be replaced, it is repaired
            last_failure_dates[
                key
            ] = date  # Record the failure date but do not replace the component

        # Append all data to lists
        dates.append(formatted_date)
        component_ids_list.append(component_id)
        component_types_list.append(component_type)
        component_ages.append(time_since_last)

        weather_temperature.append(weather_data["temperature"])
        weather_humidity.append(weather_data["humidity"])
        weather_rainfall.append(weather_data["rainfall"])
        weather_windspeed.append(weather_data["wind_speed"])
        weather_solar.append(weather_data["solar_radiation"])
        weather_snow.append(weather_data["snow"])

        traffic_load_train_size.append(traffic_load["train_size"])
        traffic_load_coach_number.append(traffic_load["coach_number"])
        traffic_load_acceleration.append(traffic_load["acceleration"])
        traffic_load_speed.append(traffic_load["speed"])
        traffic_load_train_type.append(traffic_load["train_type"])
        traffic_load_total_weight.append(traffic_load["total_weight"])  # in tons

        track_geometry_length.append(track_geometry["length"])  # in kilometers
        track_geometry_width.append(track_geometry["width"])  # in meters
        track_geometry_curvature.append(track_geometry["curvature"])
        track_geometry_gradient.append(track_geometry["gradient"])

        time_since_last_maintenance.append(time_since_last)
        number_of_maintenances.append(maintenance_data[1])
        recent_maintenance_type.append(maintenance_data[2])

        # Determine if a failure occurs
        failures.append(failure_occurred)


# Convert lists to DataFrame
df = pd.DataFrame(
    {
        "Date": dates,
        "Component": components_loop,
        "Component_ID": components_ids_loop,
        "Component_Age": component_ages,
        # Traffic data
        "Traffic_Load_Train_Size": traffic_load_train_size,  # Randomly generated in your loop
        "Traffic_Load_Coach_Number": traffic_load_coach_number,  # Randomly generated
        "Traffic_Load_Acceleration": traffic_load_acceleration,  # Randomly generated
        "Traffic_Load_Speed": traffic_load_speed,  # Randomly generated
        "Traffic_Load_Train_Type": traffic_load_train_type,  # Randomly generated
        "Traffic_Load_Total_Weight": traffic_load_total_weight,  # Randomly generated
        # Track data
        "Track_Geometry_Length": track_geometry_length,  # Randomly generated
        "Track_Geometry_Width": track_geometry_width,  # Randomly generated
        "Track_Geometry_Curvature": track_geometry_curvature,  # Randomly generated
        "Track_Geometry_Gradient": track_geometry_gradient,  # Randomly generated
        # Weather data
        "Weather_Air_Temperature": weather_temperature,  # Generated based on date
        "Weather_Relative_Humidity": weather_humidity,  # Randomly generated
        "Weather_Rainfall": weather_rainfall,  # Randomly generated
        "Weather_Wind_Speed": weather_windspeed,  # Randomly generated
        "Weather_Solar_Radiation": weather_solar,  # Randomly generated
        "Weather_Snow": weather_snow,  # Generated based on date
        # Failure
        "Failure": failures,
    }
)

df.to_csv("simulated_failure_data_2.csv", index=False)
