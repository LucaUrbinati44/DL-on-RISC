#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def pareto_front(latencies, rates):
    """Calcola il fronte di Pareto (latenza min, rate max)."""
    is_pareto = np.ones(len(latencies), dtype=bool)
    for i, (lat, rate) in enumerate(zip(latencies, rates)):
        if is_pareto[i]:
            is_pareto[is_pareto] = [
                not (latencies[j] <= lat and rates[j] >= rate and (latencies[j] < lat or rates[j] > rate))
                for j in np.where(is_pareto)[0]
            ]
            is_pareto[i] = True
    return is_pareto


def plot_pareto_scatter(files, output_png):
    """Crea uno scatter plot con fronte di Pareto da più CSV."""
    colors = plt.cm.tab10.colors
    color_map = {}
    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'v', '<', '>']
    marker_map = {}
    current_color_index = 0
    current_marker_index = 0

    fig, ax = plt.subplots(figsize=(12, 8))

    all_lat, all_rate, all_sets, all_micros = [], [], [], []

    # --- Lettura e aggregazione dati ---
    for file in files:
        df = pd.read_csv(file)
        df = df[(df['Error_does_not_fit'] == 0) & (df['Error_model_in_ram'] == 0)]

        if df.empty:
            print(f"[ATTENZIONE] Nessun dato valido in {file}")
            continue

        settings = df['end_folder_Training_Size_dd_epochs'].iloc[0]
        if settings not in color_map:
            color_map[settings] = colors[current_color_index % len(colors)]
            current_color_index += 1

        # Associa lo stesso marker al micro base
        for env in df['env_name'].unique():
            base_micro = env.replace('_modelinram', '')
            if base_micro not in marker_map:
                marker_map[base_micro] = markers[current_marker_index % len(markers)]
                current_marker_index += 1

        all_lat.extend(df['mean_tot_latency'])
        all_rate.extend(df['Rate_DL_py_load_test_tflite_mcu'])
        all_sets.extend([settings] * len(df))
        all_micros.extend(df['env_name'])

    all_lat, all_rate = np.array(all_lat), np.array(all_rate)
    all_sets, all_micros = np.array(all_sets), np.array(all_micros)

    pareto_mask = pareto_front(all_lat, all_rate)

    # --- Scatter Plot ---
    for settings in np.unique(all_sets):
        for env in np.unique(all_micros):
            mask = (all_sets == settings) & (all_micros == env)
            if not np.any(mask):
                continue

            base_micro = env.replace('_modelinram', '')
            color = color_map[settings]
            marker = marker_map[base_micro]

            # Pieno per modelinram, solo contorno per altri
            if 'modelinram' in env.lower():
                ax.scatter(all_rate[mask], all_lat[mask],
                           marker=marker, facecolor=color, edgecolor='black',
                           label=f'{settings} / {env}', s=80, alpha=0.9)
            else:
                ax.scatter(all_rate[mask], all_lat[mask],
                           marker=marker, facecolors='none', edgecolors=color,
                           label=f'{settings} / {env}', s=80, alpha=0.9)

    # --- Fronte di Pareto ---
    pareto_points = sorted([(all_rate[i], all_lat[i]) for i in np.where(pareto_mask)[0]], key=lambda x: x[0])
    pareto_points = np.array(pareto_points)
    ax.plot(pareto_points[:, 0], pareto_points[:, 1], 'k-', linewidth=2, label='Pareto front')

    # --- Legende ---
    # Micro (marker)
    handles_m = [
        plt.Line2D([0], [0], marker=m, color='w', label=micro,
                   markerfacecolor='k', markersize=10)
        for micro, m in marker_map.items()
    ]
    legend1 = ax.legend(handles=handles_m, title='Micro (marker)', loc='upper right', bbox_to_anchor=(1.28, 1))
    ax.add_artist(legend1)

    # Settings (colori)
    handles_c = [
        plt.Line2D([0], [0], marker='o', color=color, label=settings, linestyle='')
        for settings, color in color_map.items()
    ]
    legend2 = ax.legend(handles=handles_c, title='Settings (colore)', loc='upper left')

    # Pareto front
    pareto_handle = plt.Line2D([0], [0], color='k', linewidth=2, label='Pareto front')
    ax.legend(handles=[pareto_handle], loc='lower right', title='Dominanza')

    # --- Etichette e stile ---
    ax.set_xlabel('Rate_DL_py_load_test_tflite_mcu', fontsize=12)
    ax.set_ylabel('mean_tot_latency', fontsize=12)
    ax.set_title('Pareto Scatter Plot: Latenza vs Rate per micro e settings', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"[INFO] Plot salvato in {output_png}")
    plt.show()


if __name__ == "__main__":
    files = [
        'ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_nucleo-h753zi.csv',
        'ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_esp32-s2-saola-tflm.csv'
    ]

    output_png = 'pareto_plot.png'
    plot_pareto_scatter(files, output_png)
