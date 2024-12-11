import csv

def export_csv(x_data, y_data, time_data, deception_data, output_file):
    if len(x_data) == 0 or len(y_data) == 0 or len(time_data) == 0 or len(deception_data) == 0:
        return  # Ensure there is data to export

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'X Coordinate', 'Y Coordinate', 'Deception Probability'])  # Updated header
        for t, x, y, d in zip(time_data, x_data, y_data, deception_data):  # Include deception_data
            writer.writerow([t, x, y, d])
