# pump-sensor
Pump sensor data to test and learn from
# Load and analyze sensor health data for predictive maintenance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load the data
df = pd.read_csv("sensor.csv")

# Parse timestamp column
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create binary label: 1 if 'BROKEN', else 0
df['status_binary'] = df['machine_status'].apply(lambda x: 1 if x == 'BROKEN' else 0)

# Hypothetical grouping (update with real labels if known)
temperature_sensors = ['sensor_00', 'sensor_02', 'sensor_15']
pressure_sensors = ['sensor_04', 'sensor_20', 'sensor_33']
vibration_sensors = ['sensor_10', 'sensor_18', 'sensor_40']
flow_sensors = ['sensor_05', 'sensor_09', 'sensor_35']

def plot_sensor_group(sensor_list, group_name):
    plt.figure(figsize=(15, 6))
    for sensor in sensor_list:
        if sensor in df.columns:
            plt.plot(df['timestamp'], df[sensor], label=sensor, alpha=0.7)
    plt.title(f'{group_name} Sensors Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Sensor Value')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plot each sensor group over time
plot_sensor_group(temperature_sensors, "Temperature")
plot_sensor_group(pressure_sensors, "Pressure")
plot_sensor_group(vibration_sensors, "Vibration")
plot_sensor_group(flow_sensors, "Flow")

# Heatmap: mean sensor values per machine status
status_means = df.groupby('machine_status').mean(numeric_only=True)
plt.figure(figsize=(15, 10))
sns.heatmap(status_means.transpose(), cmap='coolwarm')
plt.title("Mean Sensor Values by Machine Status")
plt.ylabel("Sensors")
plt.xlabel("Machine Status")
plt.tight_layout()
plt.show()

# Correlation with failure (BROKEN status)
if 'BROKEN' in df['machine_status'].values:
    corr_data = df.drop(columns=['timestamp', 'machine_status'])
    corr = corr_data.corr(numeric_only=True)['status_binary'].drop('status_binary').sort_values(ascending=False)

    # Top correlated sensors
    plt.figure(figsize=(10, 6))
    corr.head(10).plot(kind='barh', title='Top 10 Sensors Correlated with Failure')
    plt.xlabel('Correlation with BROKEN Status')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Report
    report = {
        "Total Rows": len(df),
        "Unique Statuses": df['machine_status'].unique().tolist(),
        "Most Correlated Sensor with Failure": corr.index[0],
        "Correlation Value": corr.iloc[0],
        "Top 5 Failure-Linked Sensors": corr.head(5).to_dict()
    }
else:
    report = {
        "Total Rows": len(df),
        "Unique Statuses": df['machine_status'].unique().tolist(),
        "Note": "No 'BROKEN' status found. Cannot compute failure correlations."
    }

# Display report
print(json.dumps(report, indent=4))
