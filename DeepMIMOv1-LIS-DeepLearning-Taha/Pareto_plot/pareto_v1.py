import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def pareto_front(latencies, rates):
    is_pareto = np.ones(len(latencies), dtype=bool)
    for i, (lat, rate) in enumerate(zip(latencies, rates)):
        if is_pareto[i]:
            is_pareto[is_pareto] = [not (latencies[j] <= lat and rates[j] >= rate and (latencies[j] < lat or rates[j] > rate)) for j in np.where(is_pareto)[0]]
            is_pareto[i] = True
    return is_pareto

def plot_pareto_scatter(files, output_png):
    colors = plt.cm.tab10.colors
    color_map = {}
    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'v', '<', '>']
    marker_map = {}
    current_color_index = 0
    current_marker_index = 0

    fig, ax = plt.subplots(figsize=(12, 8))

    all_lat = []
    all_rate = []
    all_sets = []
    all_micros = []

    for file in files:
        df = pd.read_csv(file)
        df = df[(df['Error_does_not_fit'] == 0) & (df['Error_model_in_ram'] == 0)]

        settings = df['end_folder_Training_Size_dd_epochs'].iloc[0]
        if settings not in color_map:
            color_map[settings] = colors[current_color_index % len(colors)]
            current_color_index += 1

        uniques_micros = df['env_name'].unique()
        for micro in uniques_micros:
            if micro not in marker_map:
                marker_map[micro] = markers[current_marker_index % len(markers)]
                current_marker_index += 1

        all_lat.extend(df['mean_tot_latency'])
        all_rate.extend(df['Rate_DL_py_load_test_tflite_mcu'])
        all_sets.extend([settings] * len(df))
        all_micros.extend(df['env_name'])

    all_lat = np.array(all_lat)
    all_rate = np.array(all_rate)
    all_sets = np.array(all_sets)
    all_micros = np.array(all_micros)

    pareto_mask = pareto_front(all_lat, all_rate)

    for settings in np.unique(all_sets):
        for micro in np.unique(all_micros):
            mask = (all_sets == settings) & (all_micros == micro)
            ax.scatter(all_rate[mask], all_lat[mask],
                       marker=marker_map[micro],
                       color=color_map[settings],
                       label=f'{settings} / {micro}')

    pareto_points = sorted([(all_rate[i], all_lat[i]) for i in np.where(pareto_mask)[0]], key=lambda x: x[0])
    pareto_points = np.array(pareto_points)
    ax.plot(pareto_points[:, 0], pareto_points[:, 1], 'k-', linewidth=2)

    handles_m = [plt.Line2D([0], [0], marker=m, color='w', label=micro,
                            markerfacecolor='k', markersize=10) for micro, m in marker_map.items()]
    legend1 = ax.legend(handles=handles_m, title='Micro', loc='upper right', bbox_to_anchor=(1.25, 1))
    ax.add_artist(legend1)

    handles_c = [plt.Line2D([0], [0], marker='o', color=color, label=settings, linestyle='') for settings, color in color_map.items()]
    ax.legend(handles=handles_c, title='Settings (end_folder_Training_Size_dd_epochs)', loc='upper left')

    ax.set_xlabel('Rate_DL_py_load_test_tflite_mcu')
    ax.set_ylabel('mean_tot_latency')
    ax.set_title('Pareto Scatter Plot: Latenza vs Rate per micro e settings')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.show()

if __name__ == "__main__":
    
    files = ['ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_nucleo-h753zi.csv', 'ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_esp32-s2-saola-tflm.csv']

    output_png = 'pareto_plot.png'

    plot_pareto_scatter(files, output_png)