#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script per la generazione di uno scatter plot Pareto
da più file CSV, confrontando rate e latenza di diversi microcontrollori.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def compute_pareto_front(x, y):
    """Restituisce gli indici dei punti appartenenti al fronte di Pareto."""
    data = np.array(list(zip(x, y)))
    pareto_mask = np.ones(data.shape[0], dtype=bool)
    for i, point in enumerate(data):
        if pareto_mask[i]:
            pareto_mask[pareto_mask] = np.any(data[pareto_mask] > point, axis=1) | np.all(data[pareto_mask] == point, axis=1)
    return np.where(pareto_mask)[0]


def plot_pareto_from_csvs(csv_files, output_file="pareto_plot.png"):
    """Genera un unico scatter plot Pareto da una lista di file CSV."""
    plt.figure(figsize=(10, 7))

    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    marker_map = {}
    color_map = {}

    for file_idx, csv_path in enumerate(csv_files):
        df = pd.read_csv(csv_path)

        # Filtra righe con errori
        df = df[(df["Error_does_not_fit"] == 0) & (df["Error_model_in_ram"] == 0)]

        if df.empty:
            print(f"[ATTENZIONE] Nessun dato valido in {csv_path}")
            continue

        # Colori per diverse configurazioni
        setting = df["end_folder_Training_Size_dd_epochs"].iloc[0]
        if setting not in color_map:
            color_map[setting] = f"C{len(color_map)}"

        # Simboli per diversi micro
        for env in df["env_name"].unique():
            base_micro = env.replace("_modelinram", "")
            if base_micro not in marker_map:
                marker_map[base_micro] = markers[len(marker_map) % len(markers)]

        # Plot dei punti
        for env in df["env_name"].unique():
            sub = df[df["env_name"] == env]
            x = sub["Rate_DL_py_load_test_tflite_mcu"]
            y = sub["mean_tot_latency"]

            base_micro = env.replace("_modelinram", "")
            marker = marker_map[base_micro]
            color = color_map[setting]

            if "modelinram" in env.lower():
                plt.scatter(x, y, marker=marker, color=color, edgecolor='black', label=f"{env} ({setting})")
            else:
                plt.scatter(x, y, facecolors='none', edgecolors=color, marker=marker, label=f"{env} ({setting})")

        # Calcolo del fronte di Pareto
        x_all = df["Rate_DL_py_load_test_tflite_mcu"].values
        y_all = df["mean_tot_latency"].values
        pareto_idx = compute_pareto_front(x_all, y_all)
        pareto_points = df.iloc[pareto_idx].sort_values(by="Rate_DL_py_load_test_tflite_mcu")
        plt.plot(pareto_points["Rate_DL_py_load_test_tflite_mcu"],
                 pareto_points["mean_tot_latency"],
                 'k-', linewidth=1.5, label=f"Pareto {os.path.basename(csv_path)}")

    # Legende
    # Mappa simbolo -> micro
    symbol_legend = [plt.Line2D([0], [0], marker=marker_map[m], color='k', linestyle='None', label=m)
                     for m in marker_map.keys()]
    # Mappa colore -> setting
    color_legend = [plt.Line2D([0], [0], color=color_map[c], marker='o', linestyle='None', label=c)
                    for c in color_map.keys()]

    plt.xlabel("Rate (Rate_DL_py_load_test_tflite_mcu)")
    plt.ylabel("Latenza (mean_tot_latency)")
    plt.title("Pareto plot - rate vs latenza")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(handles=symbol_legend + color_legend, loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"[INFO] Plot salvato in: {output_file}")


def main():
    # Inserire qui i percorsi dei file CSV
    csv_files = [
        'ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_nucleo-h753zi.csv',
        'ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_esp32-s2-saola-tflm.csv'
    ]

    plot_pareto_from_csvs(csv_files, output_file="pareto_plot.png")


if __name__ == "__main__":
    main()
