# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 14:09:39 2025

@author: USUARIO
"""

# -*- coding: utf-8 -*-
"""
Análisis Geotécnico y Diseño Estructural de Muros de Contención - VERSIÓN FINAL CORREGIDA
Autor: Gemini
Fecha: 29 de Julio de 2025
Versión: Corregido el error en la función de actualización de datos.

Descripción:
Este programa realiza un análisis geotécnico y estructural completo. Esta versión final
corrige un bug crítico en la función get_input_data que borraba los datos existentes
al intentar editarlos. Se ha restaurado la funcionalidad de edición segura.
"""

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import datetime
import os

# --- CLASE PARA DUPLICAR LA SALIDA A CONSOLA Y ARCHIVO ---
class Logger:
    def __init__(self, filename="resultados.txt", mode='w'):
        self.terminal = sys.stdout
        self.log = open(filename, mode, encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        if self.log:
            self.log.close()
            self.log = None

# --- FUNCIÓN PARA OBTENER DATOS DE ENTRADA (CORREGIDA)---
def get_input_data(existing_data=None):
    """
    Solicita al usuario las dimensiones, parámetros del suelo y propiedades de materiales.
    CORREGIDO: Asegura que al editar (opción 3), los datos existentes se conserven.
    """
    print("\n--- Ingrese las dimensiones del muro, parámetros del suelo y materiales ---")
    
    # Lógica para determinar el modo de operación y preparar los datos
    if existing_data is None:
        user_choice_mode = input("¿Desea utilizar los valores predeterminados? (si/no): ").lower().strip()
        if user_choice_mode == 'si':
            data = { "H2": 6.0, "B": 4.0, "H3": 0.7, "D": 1.5, "t_b": 0.7, "t_c": 0.5, "p_p": 0.7, "p_t": 2.6, "cara_vertical": 'interior', "Ai": 10.0, "H1": 0.458, "D2": 0.0, "P1": 18.0, "Af1": 30.0, "C1": 0.0, "P2": 19.0, "Af2": 20.0, "C2": 40.0, "Pem": 23.58, "Pes": 18.0, "qsc": 10.0, "Kh": 0.15, "fc": 21.0, "fy": 420.0 }
            print("\nValores predeterminados cargados.")
            if data.get("cara_vertical") == 'interior': data["Bb1"], data["Bb3"] = 0.0, data["t_b"] - data["t_c"]
            else: data["Bb1"], data["Bb3"] = data["t_b"] - data["t_c"], 0.0
            data["Bb2"] = data["t_c"]
            return data
        else:
            data = {}
            user_choice_mode = '2' # Proceder a entrada manual
    else:
        print("\nHay datos existentes. ¿Qué desea hacer?")
        print("1. Cargar valores predeterminados (sobrescribir)")
        print("2. Ingresar todos los valores manualmente (sobrescribir)")
        print("3. Editar valores específicos")
        user_choice_mode = input("Ingrese su opción (1-3): ").strip()

        if user_choice_mode == '1':
            data = { "H2": 6.0, "B": 4.0, "H3": 0.7, "D": 1.5, "t_b": 0.7, "t_c": 0.5, "p_p": 0.7, "p_t": 2.6, "cara_vertical": 'interior', "Ai": 10.0, "H1": 0.458, "D2": 0.0, "P1": 18.0, "Af1": 30.0, "C1": 0.0, "P2": 19.0, "Af2": 20.0, "C2": 40.0, "Pem": 23.58, "Pes": 18.0, "qsc": 10.0, "Kh": 0.15, "fc": 21.0, "fy": 420.0 }
            print("\nValores predeterminados cargados.")
            if data.get("cara_vertical") == 'interior': data["Bb1"], data["Bb3"] = 0.0, data["t_b"] - data["t_c"]
            else: data["Bb1"], data["Bb3"] = data["t_b"] - data["t_c"], 0.0
            data["Bb2"] = data["t_c"]
            return data
        elif user_choice_mode == '2':
            data = {}
        elif user_choice_mode == '3':
            data = existing_data.copy()
        else:
            print("Opción no válida. No se han modificado los datos.")
            return existing_data

    # Campos para la entrada de datos
    input_fields = { "H2": "Altura pantalla (H2 en m): ", "B": "Ancho total zapata (B m): ", "H3": "Altura zapata (H3 m): ", "D": "Profundidad cimentación (D m): ", "t_b": "Espesor muro base (t_b m): ", "t_c": "Espesor muro corona (t_c m): ", "p_p": "Proyección pie zapata (p_p m): ", "p_t": "Proyección talón zapata (p_t m): ", "cara_vertical": "¿Cara vertical del tallo? (interior/exterior): ", "Ai": "Ángulo inclinación terreno (Ai °): ", "H1": "Altura adicional terreno inclinado (H1 m): ", "D2": "Profundidad dentellón (D2 m): ", "P1": "Peso esp. relleno (P1 kN/m³): ", "Af1": "Ángulo fricción relleno (Af1 °): ", "C1": "Cohesión relleno (C1 kPa): ", "P2": "Peso esp. cimentación (P2 kN/m³): ", "Af2": "Ángulo fricción cimentación (Af2 °): ", "C2": "Cohesión cimentación (C2 kPa): ", "Pem": "Peso esp. concreto (Pem kN/m³): ", "Pes": "Peso esp. suelo sobre zapata (Pes kN/m³): ", "qsc": "Sobrecarga relleno (qsc kPa): ", "Kh": "Coef. sísmico horizontal (Kh): ", "fc": "Resistencia del concreto f'c (MPa): ", "fy": "Fluencia del acero fy (MPa): " }

    # Bucle para solicitar datos (solo para modos 2 y 3)
    for key, prompt in input_fields.items():
        current_value = data.get(key)
        prompt_text = prompt
        is_str_option = key == "cara_vertical"
        
        if user_choice_mode == '3' and current_value is not None:
            prompt_text = f"{prompt} [Actual: {current_value}]: " if is_str_option else f"{prompt} [Actual: {current_value:.3f}]: "
        
        user_input = input(prompt_text).strip()
        
        if user_input == '' and user_choice_mode == '3':
            continue # Si se presiona Enter en modo edición, se mantiene el valor actual

        if is_str_option:
            if user_input.lower() in ['interior', 'exterior']:
                data[key] = user_input.lower()
            elif user_choice_mode != '3': # Para entrada manual, si es inválido, se pone por defecto
                 data[key] = 'interior'
        else:
            try:
                data[key] = float(user_input)
            except ValueError:
                if user_choice_mode != '3': # Para entrada manual, si es inválido, se pone 0
                    data[key] = 0.0
                # En modo edición, si es inválido, el 'continue' ya conservó el valor
    
    # Post-procesamiento final de datos
    if data.get("cara_vertical") == 'interior': data["Bb1"], data["Bb3"] = 0.0, data["t_b"] - data["t_c"]
    else: data["Bb1"], data["Bb3"] = data["t_b"] - data["t_c"], 0.0
    data["Bb2"] = data["t_c"]
    return data


# --- FUNCIÓN DE GRAFICADO ---
def graficar_muro_contencion(dims):
    # ... (código sin cambios)
    H2, B, H3, D, t_b, t_c, p_p, p_t, cara_vertical, Ai, H1_talud = (dims["H2"], dims["B"], dims["H3"], dims["D"], dims["t_b"], dims["t_c"], dims["p_p"], dims["p_t"], dims["cara_vertical"], dims["Ai"], dims["H1"])
    fig, ax = plt.subplots(figsize=(10, 8)); C_x, C_y = 0, 0; Nivel_terreno_frente = D
    zapata_poly = [(C_x, C_y), (C_x + B, C_y), (C_x + B, C_y + H3), (C_x, C_y + H3)]; ax.add_patch(patches.Polygon(zapata_poly, closed=True, linewidth=1, edgecolor='black', facecolor='lightgray', label='Zapata'))
    tallo_y_base = C_y + H3; tallo_y_corona = tallo_y_base + H2; tallo_x_base_exterior = C_x + p_p
    if cara_vertical == 'interior': puntos_tallo = [(tallo_x_base_exterior, tallo_y_base), (tallo_x_base_exterior + t_b, tallo_y_base), (tallo_x_base_exterior + t_b, tallo_y_corona), (tallo_x_base_exterior + (t_b - t_c), tallo_y_corona)]
    else: puntos_tallo = [(tallo_x_base_exterior, tallo_y_base), (tallo_x_base_exterior + t_b, tallo_y_base), (tallo_x_base_exterior + t_c, tallo_y_corona), (tallo_x_base_exterior, tallo_y_corona)]
    ax.add_patch(patches.Polygon(puntos_tallo, closed=True, linewidth=1, edgecolor='black', facecolor='gray', label='Tallo del Muro')); ax.axhline(y=Nivel_terreno_frente, color='brown', linestyle='-', linewidth=2, label='Nivel Terreno Frontal')
    x_fill_start = puntos_tallo[1][0]; Talon_fin_x = C_x + B; fill_poly_points = [(x_fill_start, tallo_y_corona), (Talon_fin_x, tallo_y_corona), (Talon_fin_x, tallo_y_base), (puntos_tallo[1][0], tallo_y_base)]
    if Ai > 0:
        x_fill_end_talud = Talon_fin_x + (H1_talud / np.tan(np.radians(Ai)) if Ai != 90 else 0); y_fill_end_talud = tallo_y_corona + H1_talud
        ax.plot([Talon_fin_x, x_fill_end_talud], [tallo_y_corona, y_fill_end_talud], color='forestgreen', linestyle='-', linewidth=2, label='Talud')
    ax.add_patch(patches.Polygon(fill_poly_points, closed=True, linewidth=0, edgecolor='none', facecolor='olivedrab', alpha=0.6, label='Suelo Relleno'))
    ax.set_aspect('equal', adjustable='box'); ax.set_title('Sección Transversal de Muro de Contención'); ax.set_xlabel('Ancho (m)'); ax.set_ylabel('Altura (m)'); ax.grid(True, linestyle='--', alpha=0.7); ax.legend(loc='upper right'); plt.tight_layout()

# --- MÓDULOS GEOTÉCNICOS Y DE DISEÑO (COMPLETOS) ---
def diseno_refuerzo(Mu, Vu, b, h, rec_cm, fc, fy):
    phi_flexion, phi_corte = 0.90, 0.85; d = h - rec_cm / 100.0
    rho_min = max(0.7 * math.sqrt(fc) / fy, 1.4 / fy) if fc > 0 and fy > 0 else 0.0018
    As_min = rho_min * b * d * 10000 if d > 0 else 0
    Vc = 0.17 * math.sqrt(fc) * b * d * 1000 if d > 0 and fc > 0 else 0; phi_Vc = phi_corte * Vc
    verificacion_corte = "Cumple" if phi_Vc >= Vu else f"NO CUMPLE (ΦVc={phi_Vc:.2f}kN < Vu={Vu:.2f}kN)"
    if d > 0 and Mu > 0 and fc > 0 and fy > 0:
        Rn_MPa = Mu / (phi_flexion * b * d**2 * 1000); term = 1 - (2 * Rn_MPa) / (0.85 * fc) if Rn_MPa <= 0.85 * fc else -1
    else: term = 1
    if term < 0: return {"Estado de la Sección": "INADECUADA: Aumentar espesor", "Momento Último (Mu) kN-m/m": Mu, "Peralte (d) cm": d * 100, "Acero Requerido (As_req) cm²/m": "> As_max (Falla)", "Acero Mínimo (As_min) cm²/m": As_min, "Acero a Usar (As_final) cm²/m": "> As_max (Falla)", "Cortante Último (Vu) kN/m": Vu, "Resistencia al Corte (ΦVc) kN": phi_Vc, "Verificación Cortante": verificacion_corte, "Refuerzo Propuesto": "N/A"}
    rho = (0.85 * fc / fy) * (1 - math.sqrt(term)) if fc > 0 and fy > 0 else 0; As_req = rho * b * d * 10000; As_final = max(As_req, As_min)
    barras_propuestas = {'1/2"': 1.29, '5/8"': 1.99, '3/4"': 2.84, '1"': 5.10}; opciones_refuerzo = []
    peralte_cm = d * 100
    if As_final > 0:
        for diam, area in barras_propuestas.items():
            s_calc = (area * 100) / As_final
            if s_calc <= peralte_cm:
                if s_calc >= 5: opciones_refuerzo.append(f"Ø {diam} @ {s_calc:.2f} cm")
    refuerzo_final_str = " ó ".join(opciones_refuerzo) if opciones_refuerzo else "Acero muy denso."
    return {"Estado de la Sección": "OK", "Momento Último (Mu) kN-m/m": Mu, "Peralte (d) cm": d * 100, "Acero Requerido (As_req) cm²/m": As_req, "Acero Mínimo (As_min) cm²/m": As_min, "Acero a Usar (As_final) cm²/m": As_final, "Cortante Último (Vu) kN/m": Vu, "Resistencia al Corte (ΦVc) kN": phi_Vc, "Verificación Cortante": verificacion_corte, "Refuerzo Propuesto": refuerzo_final_str}
def realizar_diseno_refuerzo(data, Ka, Kae, presiones_servicio):
    print("\n" + "="*85); print("                DISEÑO DE REFUERZO ESTRUCTURAL (NORMA E.060 PERÚ)                "); print("="*85)
    H2, t_b, t_c, H1, P1, Pem, qsc, Kh, fc, fy = (data[k] for k in ["H2", "t_b", "t_c", "H1", "P1", "Pem", "qsc", "Kh", "fc", "fy"]); Bt, Bp, H3, Pes, B = (data[k] for k in ["p_t", "p_p", "H3", "Pes", "B"])
    qpie_est, qtalon_est = presiones_servicio['qpie_est'], presiones_servicio['qtalon_est']; qpie_sis, qtalon_sis = presiones_servicio['qpie_sis'], presiones_servicio['qtalon_sis']
    rec_muro_cm, rec_zapata_cm = 5.0, 7.5; b_diseno = 1.0; F_D, F_L, F_D_sismo, F_L_sismo, F_E = 1.4, 1.7, 1.2, 1.0, 1.0
    formatter = {'Valor': lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x}
    print("\n--- 1. Diseño del Refuerzo de la Pantalla (Tallo) ---\n")
    secciones_y = {"Base de la Pantalla": 0, "Media Altura": H2 / 2.0, "A 2/3 de la Altura": H2 * 2.0 / 3.0}
    for nombre, y in secciones_y.items():
        if H2 <= 0: continue
        print(f"--- Análisis en: {nombre} (a {y:.2f} m de la base del tallo) ---"); espesor = t_b - (t_b - t_c) * y / H2 if H2 > 0 else t_b
        h_act, h_total = H2 - y, (H2-y) + H1; V_s, M_s = 0.5 * P1 * h_total**2 * Ka, (0.5 * P1 * h_total**2 * Ka) * (h_total / 3.0)
        V_q, M_q = qsc * h_total * Ka, (qsc * h_total * Ka) * (h_total / 2.0); Vu_e, Mu_e = F_D * V_s + F_L * V_q, F_D * M_s + F_L * M_q
        print("\n  Combinación de Carga Estática:"); print(pd.DataFrame([diseno_refuerzo(Mu_e, Vu_e, b_diseno, espesor, rec_muro_cm, fc, fy)]).T.rename(columns={0: "Valor"}).to_string(formatters=formatter))
        if Kh > 0:
            V_si, M_si = 0.5*P1*h_total**2*(Kae-Ka), (0.5*P1*h_total**2*(Kae-Ka))*(0.6*h_total); V_i, M_i = ((t_c+espesor)/2.0*h_act*Pem)*Kh, (((t_c+espesor)/2.0*h_act*Pem)*Kh)*(h_act/2.0)
            V_E, M_E = V_si + V_i, M_si + M_i; Vu_s1, Mu_s1 = F_D_sismo*V_s+F_L_sismo*V_q+F_E*V_E, F_D_sismo*M_s+F_L_sismo*M_q+F_E*M_E; Vu_s2, Mu_s2 = 0.9*V_s+F_E*V_E, 0.9*M_s+F_E*M_E
            print("\n  Combinación de Carga Sísmica:"); print(pd.DataFrame([diseno_refuerzo(max(Mu_s1,Mu_s2), max(Vu_s1,Vu_s2), b_diseno, espesor, rec_muro_cm, fc, fy)]).T.rename(columns={0: "Valor"}).to_string(formatters=formatter))
        print("-" * 60)
    print("\n\n--- 2. Diseño del Refuerzo de la Zapata ---\n"); print(f"--- Análisis del Talón (Ancho: {Bt:.2f} m) ---")
    W_suelo_t, M_suelo_t = Bt*H2*Pes, (Bt*H2*Pes)*(Bt/2.0); W_qsc_t, M_qsc_t = Bt*qsc, (Bt*qsc)*(Bt/2.0); W_zap_t, M_zap_t = Bt*H3*Pem, (Bt*H3*Pem)*(Bt/2.0)
    x_cara_tallo = Bp+t_b; q_cara_t_e = qtalon_est+(qpie_est-qtalon_est)*(B-x_cara_tallo)/B if B>0 else 0
    V_reac_e, M_reac_e = (q_cara_t_e+qtalon_est)/2.0*Bt, ((q_cara_t_e*Bt*Bt/2.0)+((qtalon_est-q_cara_t_e)*Bt/2.0)*(2.0*Bt/3.0))
    Vu_t_e, Mu_t_e = abs((F_D*(W_suelo_t+W_zap_t)+F_L*W_qsc_t)-V_reac_e), max((F_D*(M_suelo_t+M_zap_t)+F_L*M_qsc_t)-M_reac_e, 0)
    print("\n  Combinación de Carga Estática:"); print(pd.DataFrame([diseno_refuerzo(Mu_t_e, Vu_t_e, b_diseno, H3, rec_zapata_cm, fc, fy)]).T.rename(columns={0: "Valor"}).to_string(formatters=formatter))
    if Kh > 0:
        q_cara_t_s = qtalon_sis+(qpie_sis-qtalon_sis)*(B-x_cara_tallo)/B if B>0 else 0; V_reac_s, M_reac_s = (q_cara_t_s+qtalon_sis)/2.0*Bt, ((q_cara_t_s*Bt*Bt/2.0)+((qtalon_sis-q_cara_t_s)*Bt/2.0)*(2.0*Bt/3.0))
        Vu1, Mu1 = abs((F_D_sismo*(W_suelo_t+W_zap_t)+F_L_sismo*W_qsc_t)-V_reac_s), (F_D_sismo*(M_suelo_t+M_zap_t)+F_L_sismo*M_qsc_t)-M_reac_s
        Vu2, Mu2 = abs((0.9*(W_suelo_t+W_zap_t))-V_reac_s), (0.9*(M_suelo_t+M_zap_t))-M_reac_s
        print("\n  Combinación de Carga Sísmica:"); print(pd.DataFrame([diseno_refuerzo(max(Mu1,Mu2,0), max(Vu1,Vu2), b_diseno, H3, rec_zapata_cm, fc, fy)]).T.rename(columns={0: "Valor"}).to_string(formatters=formatter))
    print("-" * 60); print(f"\n--- Análisis de la Puntera (Ancho: {Bp:.2f} m) ---")
    W_zap_p, M_zap_p = Bp*H3*Pem, (Bp*H3*Pem)*(Bp/2.0); q_cara_p_e = qpie_est+(qtalon_est-qpie_est)*Bp/B if B>0 else 0
    V_reac_p_e, M_reac_p_e = (q_cara_p_e+qpie_est)/2.0*Bp, ((qpie_est*Bp*Bp/2.0)+((q_cara_p_e-qpie_est)*Bp/2.0)*(Bp/3.0))
    Vu_p_e, Mu_p_e = abs(V_reac_p_e-F_D*W_zap_p), max(M_reac_p_e-F_D*M_zap_p, 0)
    print("\n  Combinación de Carga Estática:"); print(pd.DataFrame([diseno_refuerzo(Mu_p_e, Vu_p_e, b_diseno, H3, rec_zapata_cm, fc, fy)]).T.rename(columns={0: "Valor"}).to_string(formatters=formatter))
    if Kh > 0:
        q_cara_p_s = qpie_sis+(qtalon_sis-qpie_sis)*Bp/B if B>0 else 0; V_reac_p_s, M_reac_p_s = (q_cara_p_s+qpie_sis)/2.0*Bp, ((qpie_sis*Bp*Bp/2.0)+((q_cara_p_s-qpie_sis)*Bp/2.0)*(Bp/3.0))
        Vu1, Mu1 = abs(V_reac_p_s-F_D_sismo*W_zap_p), M_reac_p_s-F_D_sismo*M_zap_p; Vu2, Mu2 = abs(V_reac_p_s-0.9*W_zap_p), M_reac_p_s-0.9*M_zap_p
        print("\n  Combinación de Carga Sísmica:"); print(pd.DataFrame([diseno_refuerzo(max(Mu1,Mu2,0), max(Vu1,Vu2), b_diseno, H3, rec_zapata_cm, fc, fy)]).T.rename(columns={0: "Valor"}).to_string(formatters=formatter))
    print("-" * 60)
def perform_geotechnical_calculations(data, method='rankine'):
    H2, B, H3, D, t_b, t_c, Bp, Bt, Ai, H1, D2, P1, Af1, C1, P2, Af2, C2, Pem, Pes, qsc, Kh = (data[k] for k in ["H2", "B", "H3", "D", "t_b", "t_c", "p_p", "p_t", "Ai", "H1", "D2", "P1", "Af1", "C1", "P2", "Af2", "C2", "Pem", "Pes", "qsc", "Kh"])
    Kv, cara_vertical = 0.0, data["cara_vertical"]; delta_wall_soil_rad = np.radians((2/3) * Af1)
    if cara_vertical == 'interior': beta_from_vertical_rad = 0.0
    else: beta_from_vertical_rad = np.arctan((t_b - t_c) / H2) if H2 > 0 else 0
    beta_from_horizontal_rad = np.radians(90.0) - beta_from_vertical_rad; Hp, Af1r, alpha_rad, Af2r_calc = H1 + H2 + H3, np.radians(Af1), np.radians(Ai), np.radians(Af2)
    Area_tallo_rect = t_c * H2; Area_tallo_tri = 0.5 * (t_b - t_c) * H2; P_tallo_rect, P_tallo_tri = Area_tallo_rect * Pem, Area_tallo_tri * Pem
    P_zapata = (B*H3) * Pem; P_suelo_talon = (Bt*H2) * Pes; P_suelo_talud = (0.5*H1*(H1/np.tan(alpha_rad)) if alpha_rad > 0 else 0)*Pes
    if cara_vertical == 'interior': Bm_tallo_rect, Bm_tallo_tri = Bp + t_c / 2.0, Bp + t_c + (t_b - t_c) / 3.0
    else: Bm_tallo_rect, Bm_tallo_tri = Bp + (t_b - t_c) + t_c/2, Bp + (2/3) * (t_b - t_c)
    Bm_zapata, Bm_suelo_talon = B/2, B - Bt/2; Bm_suelo_talud = B - (H1/np.tan(alpha_rad))/3 if alpha_rad > 0 else B - Bt
    M_tallo_rect, M_tallo_tri = P_tallo_rect*Bm_tallo_rect, P_tallo_tri*Bm_tallo_tri; M_zapata, M_suelo_talon, M_suelo_talud = P_zapata*Bm_zapata, P_suelo_talon*Bm_suelo_talon, P_suelo_talud*Bm_suelo_talud
    SFv_estatico_R = P_tallo_rect + P_tallo_tri + P_zapata + P_suelo_talon + P_suelo_talud; SMr_estatico_R = M_tallo_rect + M_tallo_tri + M_zapata + M_suelo_talon + M_suelo_talud
    if method == 'rankine':
        term1_Ka = np.cos(alpha_rad)
        if np.cos(alpha_rad)**2 < np.cos(Af1r)**2: Ka = 1.0 
        else: term2_Ka = np.sqrt(np.cos(alpha_rad)**2 - np.cos(Af1r)**2); Ka = term1_Ka * (term1_Ka - term2_Ka) / (term1_Ka + term2_Ka)
        Pa_estatico = 0.5 * P1 * Hp**2 * Ka; Pv_estatico, Ph_estatico = Pa_estatico*np.sin(alpha_rad), Pa_estatico*np.cos(alpha_rad); P_qsc_h_estatico = qsc * Ka * Hp * np.cos(alpha_rad)
    else: # coulomb
        try:
            num_Ka = np.sin(beta_from_horizontal_rad + Af1r)**2; den1_Ka = np.sin(beta_from_horizontal_rad)**2 * np.sin(beta_from_horizontal_rad - delta_wall_soil_rad)
            sqrt_num = np.sin(Af1r + delta_wall_soil_rad) * np.sin(Af1r - alpha_rad); sqrt_den = np.sin(beta_from_horizontal_rad - delta_wall_soil_rad) * np.sin(alpha_rad + beta_from_horizontal_rad)
            if den1_Ka <= 0 or sqrt_den <= 0 or (sqrt_num / sqrt_den) < 0: Ka = 1.0
            else: den2_Ka = (1 + np.sqrt(sqrt_num / sqrt_den))**2; Ka = num_Ka / (den1_Ka * den2_Ka)
        except (ValueError, ZeroDivisionError): Ka = 1.0
        Pa_estatico = 0.5 * P1 * Hp**2 * Ka; Ph_estatico = Pa_estatico * np.cos(delta_wall_soil_rad + np.radians(90) - beta_from_horizontal_rad); Pv_estatico = Pa_estatico * np.sin(delta_wall_soil_rad + np.radians(90) - beta_from_horizontal_rad); P_qsc_h_estatico = qsc * Hp * Ka * np.cos(delta_wall_soil_rad + np.radians(90) - beta_from_horizontal_rad)
    SFv_estatico_total = SFv_estatico_R + Pv_estatico; SMr_estatico_total = SMr_estatico_R + (Pv_estatico * B); Mo_total_estatico = Ph_estatico * (Hp/3) + P_qsc_h_estatico * (Hp/2); FSvolteo_estatico = SMr_estatico_total/Mo_total_estatico if Mo_total_estatico > 0 else float('inf')
    Dp, Kp = D+D2, np.tan(np.pi/4+np.radians(Af2)/2)**2; Pp_estatico = 0.5*Kp*P2*Dp**2+2*C2*np.sqrt(Kp)*Dp; den_fs_desl_est = Ph_estatico+P_qsc_h_estatico; FSdeslizamiento_estatico = (SFv_estatico_total*np.tan(2/3*Af2r_calc) + B*2/3*C2 + Pp_estatico)/den_fs_desl_est if den_fs_desl_est > 0 else float('inf')
    e_estatico = B/2 - (SMr_estatico_total - Mo_total_estatico)/SFv_estatico_total if SFv_estatico_total > 0 else 0; emax_estatico = B/6
    qpie_estatico, qtalon_estatico = (SFv_estatico_total/B)*(1+6*e_estatico/B) if B>0 else 0, (SFv_estatico_total/B)*(1-6*e_estatico/B) if B>0 else 0; Bp_eff_estatico = B - 2 * e_estatico
    if Af2 == 20: Nq, Nc, Ny = 6.4, 14.83, 5.39
    else: Nq = np.tan(np.radians(45+Af2/2))**2 * np.exp(np.pi*np.tan(Af2r_calc)); Nc = (Nq-1)/np.tan(Af2r_calc) if Af2r_calc > 0 else 5.14; Ny = 2*(Nq+1)*np.tan(Af2r_calc) if Af2r_calc > 0 else 0
    Fcd, Fqd, Fyd = 1+0.4*(D/Bp_eff_estatico) if Bp_eff_estatico>0 else 1, 1+2*np.tan(Af2r_calc)*(1-np.sin(Af2r_calc))**2*(D/Bp_eff_estatico) if Bp_eff_estatico>0 else 1, 1
    Y_est = np.arctan((Ph_estatico+P_qsc_h_estatico)/SFv_estatico_total) if SFv_estatico_total>0 else 0; Fci, Fqi, Fyi = (1-Y_est/(np.pi/2))**2, (1-Y_est/(np.pi/2))**2, (1-Y_est/Af2r_calc)**2 if Af2r_calc>0 else 0
    qu_estatico = C2*Nc*Fcd*Fci + P2*D*Nq*Fqd*Fqi + 0.5*P2*Bp_eff_estatico*Ny*Fyd*Fyi
    FScarga_estatico = qu_estatico/qpie_estatico if qpie_estatico > 0 else float('inf')
    Kae = Ka; Pae_seismic, Delta_Ph, Delta_Pv, P_Iw, P_Is = 0,0,0,0,0; Mo_Delta_Ph, Mo_Iw, Mo_Is, Mo_total_seismic = 0,0,0,0
    Ph_seismic_total, Pv_seismic_total = 0, 0; e_seismic, qpie_seismic, qtalon_seismic, qu_seismic = e_estatico, qpie_estatico, qtalon_estatico, qu_estatico
    FSvolteo_seismic, FSdeslizamiento_seismic, FScarga_seismic = np.nan, np.nan, np.nan
    if Kh > 0:
        theta_w_rad = np.arctan(Kh/(1-Kv))
        try:
            num_Kae = np.cos(Af1r - theta_w_rad - beta_from_vertical_rad)**2; den1_Kae = np.cos(theta_w_rad) * np.cos(beta_from_vertical_rad)**2 * np.cos(delta_wall_soil_rad + beta_from_vertical_rad + theta_w_rad)
            sqrt_num_s = np.sin(Af1r + delta_wall_soil_rad) * np.sin(Af1r - theta_w_rad - alpha_rad); sqrt_den_s = np.cos(delta_wall_soil_rad + beta_from_vertical_rad + theta_w_rad) * np.cos(alpha_rad - beta_from_vertical_rad)
            if den1_Kae <= 0 or sqrt_den_s <= 0 or (sqrt_num_s / sqrt_den_s) < 0: Kae = Ka
            else: den2_Kae = (1 + np.sqrt(sqrt_num_s / sqrt_den_s))**2; Kae = num_Kae / (den1_Kae * den2_Kae)
        except (ValueError, ZeroDivisionError): Kae = Ka
        Pae_seismic = 0.5 * P1 * Hp**2 * Kae * (1-Kv)
        if method == 'rankine': Ph_seismic_total = Pae_seismic * np.cos(alpha_rad); Pv_seismic_total = Pae_seismic * np.sin(alpha_rad)
        else: Ph_seismic_total = Pae_seismic * np.cos(delta_wall_soil_rad + np.radians(90) - beta_from_horizontal_rad); Pv_seismic_total = Pae_seismic * np.sin(delta_wall_soil_rad + np.radians(90) - beta_from_horizontal_rad)
        Delta_Ph = Ph_seismic_total - Ph_estatico; Delta_Pv = Pv_seismic_total - Pv_estatico; P_Iw = (P_tallo_rect+P_tallo_tri+P_zapata)*Kh; P_Is = (P_suelo_talon+P_suelo_talud)*Kh
        SFv_seismic_total = SFv_estatico_R + Pv_seismic_total; SMr_seismic_total = SMr_estatico_R + (Pv_seismic_total * B); Mo_Delta_Ph = Delta_Ph * (0.6 * Hp); CG_muro_y = H3 + (H2 / 2) if H2 > 0 else H3; Mo_Iw = P_Iw * CG_muro_y
        CG_suelo_y = H3 + (H2 / 2) if H2 > 0 else H3; Mo_Is = P_Is * CG_suelo_y; Mo_total_seismic = Mo_total_estatico + Mo_Delta_Ph + Mo_Iw + Mo_Is
        FSvolteo_seismic = SMr_seismic_total / Mo_total_seismic if Mo_total_seismic > 0 else float('inf'); Ph_total_seismic_desl = Ph_estatico + P_qsc_h_estatico + Delta_Ph + P_Iw + P_Is
        FSdeslizamiento_seismic = (SFv_seismic_total*np.tan(2/3*Af2r_calc) + B*2/3*C2 + Pp_estatico) / Ph_total_seismic_desl if Ph_total_seismic_desl > 0 else float('inf')
        e_seismic = B/2 - (SMr_seismic_total-Mo_total_seismic) / SFv_seismic_total if SFv_seismic_total > 0 else 0
        qpie_seismic, qtalon_seismic = (SFv_seismic_total/B)*(1+6*e_seismic/B) if B>0 else 0, (SFv_seismic_total/B)*(1-6*e_seismic/B) if B>0 else 0; Bp_eff_seismic = B - 2*e_seismic
        Y_sis = np.arctan(Ph_total_seismic_desl/SFv_seismic_total) if SFv_seismic_total>0 else 0; Fci_s, Fqi_s, Fyi_s = (1-Y_sis/(np.pi/2))**2, (1-Y_sis/(np.pi/2))**2, (1-Y_sis/Af2r_calc)**2 if Af2r_calc>0 else 0
        qu_seismic = C2*Nc*Fcd*Fci_s + P2*D*Nq*Fqd*Fqi_s + 0.5*P2*Bp_eff_seismic*Ny*Fyd*Fyi_s
        FScarga_seismic = qu_seismic/qpie_seismic if qpie_seismic > 0 else float('inf')
    results = locals(); return results

# --- MÓDULOS DE IMPRESIÓN Y GRAFICADO GEOTÉCNICO (RESTAURADOS) ---
def print_geotechnical_report(results, method_name):
    data = results['data']; Ka = results['Ka']; Ph_estatico = results['Ph_estatico']; P_qsc_h_estatico = results['P_qsc_h_estatico']
    FSvolteo_estatico = results['FSvolteo_estatico']; FSdeslizamiento_estatico = results['FSdeslizamiento_estatico']
    FScarga_estatico = results['FScarga_estatico']; e_estatico = results['e_estatico']; emax_estatico = results['emax_estatico']
    qpie_estatico = results['qpie_estatico']; qtalon_estatico = results['qtalon_estatico']; qu_estatico = results['qu_estatico']
    Kh = data['Kh']
    print("\n" + "="*80); print(f"           ANÁLISIS ESTÁTICO DEL MURO DE CONTENCIÓN - MÉTODO DE {method_name.upper()}         "); print("="*80)
    print(f"\nDimensiones y Parámetros del Muro y Suelo:"); print(pd.DataFrame(list(data.items()), columns=['Parámetro', 'Valor']).to_string(index=False))
    print(f"\n--- Empujes y Sobrecargas (Estático) ---"); print(f"  Coeficiente de Presión Activa (Ka): {Ka:.3f}"); print(f"  Componente Horizontal de Pa (Ph): {Ph_estatico:.2f} kN/m"); print(f"  Fuerza Horizontal por Sobrecarga (P_qsc): {P_qsc_h_estatico:.2f} kN/m")
    print(f"\n--- Verificación de Estabilidad (Estático) ---")
    verif_data_est = {"Verificación": ["Volteo", "Deslizamiento", "Capacidad de Carga", "Excentricidad"], "Valor Calculado": [f"{FSvolteo_estatico:.2f}", f"{FSdeslizamiento_estatico:.2f}", f"{FScarga_estatico:.2f}", f"{e_estatico:.3f} m"], "Valor Requerido/Límite": [">= 1.5", ">= 1.5", ">= 3.0", f"<= {emax_estatico:.3f} m (B/6)"], "Estado": ["✅ Cumple" if FSvolteo_estatico >= 1.5 else "❌ No Cumple", "✅ Cumple" if FSdeslizamiento_estatico >= 1.5 else "❌ No Cumple", "✅ Cumple" if FScarga_estatico >= 3.0 else "❌ No Cumple", "✅ Cumple" if abs(e_estatico) <= emax_estatico else "❌ No Cumple"]}
    print(pd.DataFrame(verif_data_est).to_string(index=False)); print(f"  Presión en el pie (q_pie): {qpie_estatico:.2f} kPa"); print(f"  Presión en el talón (q_talón): {qtalon_estatico:.2f} kPa"); print(f"  Capacidad de Carga Última (qu): {qu_estatico:.2f} kPa")
    if Kh > 0 and 'FSvolteo_seismic' in results and not np.isnan(results['FSvolteo_seismic']):
        FSvolteo_seismic, FSdeslizamiento_seismic, FScarga_seismic = results['FSvolteo_seismic'], results['FSdeslizamiento_seismic'], results['FScarga_seismic']
        e_seismic, qpie_seismic, qtalon_seismic, qu_seismic = results['e_seismic'], results['qpie_seismic'], results['qtalon_seismic'], results['qu_seismic']
        emax_seismic = data['B'] / 4
        print("\n" + "="*80); print("                   ANÁLISIS SÍSMICO DEL MURO DE CONTENCIÓN (MONONOBE-OKABE)                "); print("="*80)
        verif_data_sis = {"Verificación": ["Volteo", "Deslizamiento", "Capacidad de Carga", "Excentricidad"], "Valor Calculado": [f"{FSvolteo_seismic:.2f}", f"{FSdeslizamiento_seismic:.2f}", f"{FScarga_seismic:.2f}", f"{e_seismic:.3f} m"], "Valor Requerido/Límite": [">= 1.2", ">= 1.25", ">= 2.0", f"<= {emax_seismic:.3f} m (B/4)"], "Estado": ["✅ Cumple" if FSvolteo_seismic >= 1.2 else "❌ No Cumple", "✅ Cumple" if FSdeslizamiento_seismic >= 1.25 else "❌ No Cumple", "✅ Cumple" if FScarga_seismic >= 2.0 else "❌ No Cumple", "✅ Cumple" if abs(e_seismic) <= emax_seismic else "❌ No Cumple"]}
        print(pd.DataFrame(verif_data_sis).to_string(index=False)); print(f"  Presión en el pie (q_pie): {qpie_seismic:.2f} kPa"); print(f"  Presión en el talón (q_talón): {qtalon_seismic:.2f} kPa"); print(f"  Capacidad de Carga Última (qu): {qu_seismic:.2f} kPa")
def plot_earth_pressures(ax, results, method_name):
    Hp, P1, qsc, Ka, Kae, Kh = (results[k] for k in ['Hp', 'P1', 'qsc', 'Ka', 'Kae', 'Kh']); Ph_estatico, P_qsc_h_estatico, Ph_seismic_total = (results[k] for k in ['Ph_estatico', 'P_qsc_h_estatico', 'Ph_seismic_total'])
    y = np.linspace(0, Hp, 101); p_qsc_h = P_qsc_h_estatico / Hp if Hp > 0 else 0; p_suelo_h_base = (Ph_estatico * 2) / Hp if Hp > 0 else 0
    presion_estatica = p_qsc_h + np.linspace(0, p_suelo_h_base, 101)
    ax.plot(presion_estatica, y, 'b-', label='Empuje Estático'); ax.fill_betweenx(y, 0, presion_estatica, color='blue', alpha=0.2)
    if Kh > 0 and Ph_seismic_total > 0:
        presion_sismica_total_base = (Ph_seismic_total * 2) / Hp if Hp > 0 else 0
        presion_sismica = p_qsc_h + np.linspace(0, presion_sismica_total_base, 101)
        ax.plot(presion_sismica, y, 'r--', label='Empuje Sísmico Total')
        ax.fill_betweenx(y, presion_estatica, presion_sismica, color='red', alpha=0.2, hatch='//')
    ax.set_title(f'Empujes - Método de {method_name}'); ax.set_xlabel('Presión Horizontal (kPa)'); ax.set_ylabel('Altura desde la base (m)'); ax.grid(True, linestyle=':'); ax.legend(); ax.invert_xaxis(); ax.yaxis.tick_right()
def plot_stem_forces(ax_v, ax_m, results, method_name):
    H1, H2, P1, qsc, Ka, Kae, Kh = (results[k] for k in ['H1', 'H2', 'P1', 'qsc', 'Ka', 'Kae', 'Kh'])
    y = np.linspace(0, H2, 101); V_est, M_est, V_sis, M_sis = [], [], [], []
    for y_i in y:
        h_activa = H1 + H2 - y_i
        V_est.append((qsc * h_activa * Ka) + (0.5 * P1 * h_activa**2 * Ka)); M_est.append((qsc * h_activa * Ka * h_activa / 2) + (0.5 * P1 * h_activa**2 * Ka * h_activa / 3))
        if Kh > 0:
            V_sis.append((qsc * h_activa * Kae) + (0.5 * P1 * h_activa**2 * Kae)); M_sis.append((qsc * h_activa * Kae * h_activa / 2) + (0.5 * P1 * h_activa**2 * Kae * h_activa / 3))
    ax_v.plot(V_est, y, 'b-', label='Cortante Estático'); ax_m.plot(M_est, y, 'g-', label='Momento Estático')
    if Kh > 0 and V_sis: ax_v.plot(V_sis, y, 'r--', label='Cortante Sísmico'); ax_m.plot(M_sis, y, 'm--', label='Momento Sísmico')
    ax_v.set_title(f'Cortante en Pantalla ({method_name})'); ax_v.set_xlabel('Cortante (kN/m)'); ax_v.set_ylabel('Altura en Pantalla (m)'); ax_v.grid(True, linestyle=':'); ax_v.legend()
    ax_m.set_title(f'Momento en Pantalla ({method_name})'); ax_m.set_xlabel('Momento (kN-m/m)'); ax_m.set_ylabel('Altura en Pantalla (m)'); ax_m.grid(True, linestyle=':'); ax_m.legend(); ax_m.invert_xaxis()
def generate_all_diagrams(results_rankine, results_coulomb):
    fig_p, (ax_pr, ax_pc) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    plot_earth_pressures(ax_pr, results_rankine, 'Rankine'); plot_earth_pressures(ax_pc, results_coulomb, 'Coulomb')
    fig_p.suptitle('Diagrama de Empuje Horizontal del Terreno', fontsize=16); fig_p.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_s, axes_s = plt.subplots(2, 2, figsize=(14, 10), sharey='row')
    plot_stem_forces(axes_s[0, 0], axes_s[1, 0], results_rankine, 'Rankine'); plot_stem_forces(axes_s[0, 1], axes_s[1, 1], results_coulomb, 'Coulomb')
    fig_s.suptitle('Diagramas de Cortante y Momento en la Pantalla (Tallo)', fontsize=16); fig_s.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ==============================================================================
# --- INICIO DEL MÓDULO DE ANÁLISIS POR ELEMENTOS FINITOS (FEM) ---
# ==============================================================================
def fem_generar_malla(data, mesh_param):
    H_pantalla = data['H2']; H_zapata = data['H3']; B_zapata = data['B']; p_p = data['p_p']; t_b = data['t_b']; t_c = data['t_c']; cara_vertical = data['cara_vertical']
    x_front_base = p_p; x_back_base = p_p + t_b
    if cara_vertical == 'interior': x_front_top = p_p + (t_b - t_c); x_back_top = p_p + t_b
    else: x_front_top = p_p; x_back_top = p_p + t_c
    div_y_zapata = int(H_zapata * mesh_param) + 1; div_y_tallo = int(H_pantalla * mesh_param) + 1
    y_coords_zapata = np.linspace(0, H_zapata, div_y_zapata); y_coords_tallo = np.linspace(H_zapata, H_zapata + H_pantalla, div_y_tallo)
    all_nodes = {}; elementos = []; node_id_counter = 1
    def mesh_rectangle(x_start, x_end, y_coords_section, store_top_nodes=False):
        nonlocal node_id_counter; num_div_x = int((x_end - x_start) * mesh_param) + 1
        if num_div_x <= 1: num_div_x = 2
        x_section_coords = np.linspace(x_start, x_end, num_div_x)
        node_grid, top_nodes_ids = {}, []
        for j, y in enumerate(y_coords_section):
            for i, x in enumerate(x_section_coords):
                coord = (round(x, 6), round(y, 6))
                if coord not in all_nodes: all_nodes[coord] = node_id_counter; node_id_counter += 1
                node_grid[(i, j)] = all_nodes[coord]
        if store_top_nodes: top_nodes_ids = [node_grid[(i, len(y_coords_section)-1)] for i in range(len(x_section_coords))]
        for j in range(len(y_coords_section) - 1):
            for i in range(len(x_section_coords) - 1):
                n1, n2, n3, n4 = node_grid.get((i,j)), node_grid.get((i+1,j)), node_grid.get((i,j+1)), node_grid.get((i+1,j+1))
                if all([n1, n2, n3, n4]): elementos.append([n1, n2, n4]); elementos.append([n1, n4, n3])
        return top_nodes_ids
    mesh_rectangle(0, p_p, y_coords_zapata)
    base_nodes_tallo_ids = mesh_rectangle(p_p, p_p + t_b, y_coords_zapata, store_top_nodes=True)
    if B_zapata > p_p + t_b: mesh_rectangle(p_p + t_b, B_zapata, y_coords_zapata)
    num_div_x_tallo = len(base_nodes_tallo_ids)
    node_grid_tallo = {(i, 0): node_id for i, node_id in enumerate(base_nodes_tallo_ids)}
    for j, y in enumerate(y_coords_tallo[1:], 1):
        interp_factor = (y - H_zapata) / H_pantalla if H_pantalla > 0 else 0
        x_start = x_front_base + (x_front_top - x_front_base) * interp_factor; x_end = x_back_base + (x_back_top - x_back_base) * interp_factor
        x_section_coords = np.linspace(x_start, x_end, num_div_x_tallo)
        for i, x in enumerate(x_section_coords):
            coord = (round(x, 6), round(y, 6))
            if coord not in all_nodes: all_nodes[coord] = node_id_counter; node_id_counter += 1
            node_grid_tallo[(i,j)] = all_nodes[coord]
    for j in range(len(y_coords_tallo) - 1):
        for i in range(num_div_x_tallo - 1):
            n1, n2, n3, n4 = node_grid_tallo[(i,j)], node_grid_tallo[(i+1,j)], node_grid_tallo[(i,j+1)], node_grid_tallo[(i+1,j+1)]
            elementos.append([n1, n2, n4]); elementos.append([n1, n4, n3])
    nudos_list = sorted(all_nodes.items(), key=lambda item: item[1])
    nudos_coords_df = pd.DataFrame([coord for coord, id in nudos_list], columns=['x', 'y']); nudos_coords_df['nudo'] = [id for coord, id in nudos_list]
    elementos_df = pd.DataFrame(elementos, columns=['n1', 'n2', 'n3']); elementos_df.index.name = 'elemento'; elementos_df.reset_index(inplace=True); elementos_df['elemento'] += 1
    nudos_base = nudos_coords_df[np.isclose(nudos_coords_df['y'], 0)].copy()
    perimetro_muro = np.array([[0,0], [B_zapata,0], [B_zapata,H_zapata], [x_back_base,H_zapata], [x_back_top,H_zapata+H_pantalla], [x_front_top,H_zapata+H_pantalla], [x_front_base,H_zapata], [0,H_zapata], [0,0]])
    return nudos_coords_df, elementos_df, nudos_base, perimetro_muro
def fem_aplicar_empuje(nudos, data, geo_results, load_case='static'):
    if load_case == 'seismic': total_force_kn = geo_results['Ph_seismic_total']; Ka_eff = geo_results['Kae']
    else: total_force_kn = geo_results['Ph_estatico'] + geo_results['P_qsc_h_estatico']; Ka_eff = geo_results['Ka']
    total_force_kgf = -total_force_kn * 101.972
    H_zapata = data['H3']; stem_nodes = nudos[nudos['y'] > H_zapata + 1e-9].copy()
    if stem_nodes.empty: return pd.DataFrame(np.zeros((len(nudos), 2)), columns=['fx', 'fy'], index=nudos['nudo']).reset_index()
    back_face_nodes_idx = stem_nodes.loc[stem_nodes.groupby('y')['x'].idxmax()].index
    nodes_on_face = stem_nodes.loc[back_face_nodes_idx].copy()
    if nodes_on_face.empty: return pd.DataFrame(np.zeros((len(nudos), 2)), columns=['fx', 'fy'], index=nudos['nudo']).reset_index()
    P1 = data['P1']; qsc = data['qsc']; H_total_presion = geo_results['Hp']
    pressures = [(qsc * Ka_eff) + (P1 * (H_total_presion - node['y']) * Ka_eff) for _, node in nodes_on_face.iterrows()]
    nodes_on_face['pressure'] = pressures; total_pressure_weight = nodes_on_face['pressure'].sum()
    if total_pressure_weight > 1e-9: nodes_on_face['force_fx'] = total_force_kgf * (nodes_on_face['pressure'] / total_pressure_weight)
    else: nodes_on_face['force_fx'] = total_force_kgf / len(nodes_on_face) if len(nodes_on_face) > 0 else 0
    fexterna = pd.DataFrame(np.zeros((len(nudos), 2)), columns=['fx', 'fy'], index=nudos['nudo'])
    for _, node in nodes_on_face.iterrows(): fexterna.loc[int(node['nudo']), 'fx'] = node['force_fx']
    return fexterna.reset_index()
def fem_matriz_rigidez_elemento(E, v, th, nudos_elem):
    C = (E / (1 - v**2)) * np.array([[1, v, 0], [v, 1, 0], [0, 0, (1 - v) / 2]]); p1, p2, p3 = nudos_elem[['x', 'y']].values
    A = 0.5 * np.linalg.det(np.array([[p1[0], p1[1], 1], [p2[0], p2[1], 1], [p3[0], p3[1], 1]]));
    if abs(A) < 1e-12: return np.zeros((6, 6))
    b1,b2,b3=p2[1]-p3[1],p3[1]-p1[1],p1[1]-p2[1]; c1,c2,c3=p3[0]-p2[0],p1[0]-p3[0],p2[0]-p1[0]
    B=(1/(2*A))*np.array([[b1,0,b2,0,b3,0],[0,c1,0,c2,0,c3],[c1,b1,c2,b2,c3,b3]]); return th*A*(B.T@C@B)
def fem_calcular_esfuerzos_elemento(E, v, nudos_elem, despl_elem):
    C = (E / (1 - v**2)) * np.array([[1, v, 0], [v, 1, 0], [0, 0, (1 - v) / 2]]); p1, p2, p3 = nudos_elem[['x', 'y']].values
    A = 0.5 * np.linalg.det(np.array([[p1[0],p1[1],1],[p2[0],p2[1],1],[p3[0],p3[1],1]]))
    if abs(A) < 1e-12: return np.zeros(3)
    b1,b2,b3=p2[1]-p3[1],p3[1]-p1[1],p1[1]-p2[1]; c1,c2,c3=p3[0]-p2[0],p1[0]-p3[0],p2[0]-p1[0]
    B=(1/(2*A))*np.array([[b1,0,b2,0,b3,0],[0,c1,0,c2,0,c3],[c1,b1,c2,b2,c3,b3]])
    deformaciones = B @ despl_elem.flatten(); esfuerzos = C @ deformaciones; return esfuerzos
def fem_visualizar_resultados(nudos, elementos, desplazamientos, E_m2, v, perimetro, escala_visual=50):
    esfuerzos_sx_lista, esfuerzos_sy_lista = [], []
    for _, elem in elementos.iterrows():
        n_ids = elem[['n1', 'n2', 'n3']].astype(int).values; nudos_elem=nudos[nudos['nudo'].isin(n_ids)].set_index('nudo').loc[n_ids]
        idx_despl=np.array([[2*(n-1),2*(n-1)+1] for n in n_ids]).flatten()
        sx, sy, txy = fem_calcular_esfuerzos_elemento(E_m2,v,nudos_elem,desplazamientos[idx_despl])
        esfuerzos_sx_lista.append(sx); esfuerzos_sy_lista.append(sy)
    elementos['Sxx'] = esfuerzos_sx_lista; elementos['Syy'] = esfuerzos_sy_lista
    plt.figure(figsize=(10, 8)); plt.gca().set_aspect('equal', adjustable='box'); plt.plot(perimetro[:, 0], perimetro[:, 1], 'k-', lw=1, label='Contorno Original')
    x_def = nudos['x'] + desplazamientos[0::2] * escala_visual; y_def = nudos['y'] + desplazamientos[1::2] * escala_visual
    plt.triplot(x_def, y_def, elementos[['n1','n2','n3']].values-1,'r-',label=f'Deformada (escala x{escala_visual})',lw=0.7)
    plt.title('Resultados FEM: Malla Deformada',fontsize=16); plt.xlabel('X (m)'); plt.ylabel('Y (m)'); plt.legend(); plt.grid(True); plt.show()
    def plot_esfuerzos(valores_esfuerzo, titulo):
        plt.figure(figsize=(10, 8)); plt.gca().set_aspect('equal', adjustable='box')
        valores_cm2 = np.array(valores_esfuerzo) / 10000; lim_max = np.abs(valores_cm2).max()
        if lim_max < 1e-9: lim_max = 1.0
        triang=plt.tripcolor(nudos['x'],nudos['y'],elementos[['n1','n2','n3']].values-1, facecolors=valores_cm2, cmap='jet', vmin=-lim_max, vmax=lim_max)
        plt.colorbar(triang,label='Esfuerzo (kg/cm²)\n(Azul: Compresión, Rojo: Tensión)'); plt.plot(perimetro[:, 0], perimetro[:, 1], 'k-', lw=0.5, alpha=0.8)
        plt.title(titulo,fontsize=16); plt.xlabel('X (m)'); plt.ylabel('Y (m)'); plt.show()
    plot_esfuerzos(elementos['Sxx'], 'Resultados FEM: Esfuerzos Horizontales ($S_{xx}$)')
    plot_esfuerzos(elementos['Syy'], 'Resultados FEM: Esfuerzos Verticales ($S_{yy}$)')
    return elementos[['Sxx', 'Syy']]
def export_fem_results_to_file(logger, data, nudos, desplazamientos_df, elementos, esfuerzos_df, method, load_case):
    logger.write("\n" + "="*85 + "\n"); logger.write(f"     RESULTADOS DEL ANÁLISIS POR ELEMENTOS FINITOS (FEM)\n")
    logger.write(f"     Método de Carga: {method.upper()} | Caso: {'SÍSMICO' if load_case == 'seismic' else 'ESTÁTICO'}\n"); logger.write("="*85 + "\n\n")
    logger.write("--- Desplazamientos Críticos ---\n")
    H_total = data['H3'] + data['H2']; p_p = data['p_p']; t_b = data['t_b']
    nudos_corona = nudos[nudos['y'] > H_total - 1e-6]; nudo_corona = nudos.loc[nudos_corona.index[len(nudos_corona)//2]]
    despl_corona = desplazamientos_df[desplazamientos_df['nudo'] == nudo_corona['nudo']]
    logger.write(f"  - Corona del Tallo (aprox. x={nudo_corona['x']:.2f}, y={nudo_corona['y']:.2f} m):\n")
    logger.write(f"    - Desplazamiento Horizontal (dx): {despl_corona['dx_mm'].iloc[0]:.4f} mm\n"); logger.write(f"    - Desplazamiento Vertical (dy):   {despl_corona['dy_mm'].iloc[0]:.4f} mm\n\n")
    nudos_union = nudos[np.isclose(nudos['y'], data['H3']) & (nudos['x'] >= p_p) & (nudos['x'] <= p_p + t_b)]
    nudo_union = nudos.loc[nudos_union.index[len(nudos_union)//2]]
    despl_union = desplazamientos_df[desplazamientos_df['nudo'] == nudo_union['nudo']]
    logger.write(f"  - Unión Tallo-Zapata (aprox. x={nudo_union['x']:.2f}, y={nudo_union['y']:.2f} m):\n")
    logger.write(f"    - Desplazamiento Horizontal (dx): {despl_union['dx_mm'].iloc[0]:.4f} mm\n"); logger.write(f"    - Desplazamiento Vertical (dy):   {despl_union['dy_mm'].iloc[0]:.4f} mm\n\n")
    despl_max_asentamiento = desplazamientos_df.loc[desplazamientos_df['dy_mm'].idxmin()]
    nudo_asentamiento = nudos[nudos['nudo'] == despl_max_asentamiento['nudo']]
    logger.write(f"  - Máximo Asentamiento en la Base (en x={nudo_asentamiento['x'].iloc[0]:.2f}, y=0.00 m):\n"); logger.write(f"    - Desplazamiento Vertical (dy):   {despl_max_asentamiento['dy_mm']:.4f} mm\n\n")
    logger.write("--- Esfuerzos Críticos (por elemento) ---\n")
    def get_centroid(elem_row):
        n1_coord = nudos[nudos['nudo'] == elem_row['n1']].iloc[0]; n2_coord = nudos[nudos['nudo'] == elem_row['n2']].iloc[0]; n3_coord = nudos[nudos['nudo'] == elem_row['n3']].iloc[0]
        return (n1_coord['x'] + n2_coord['x'] + n3_coord['x']) / 3, (n1_coord['y'] + n2_coord['y'] + n3_coord['y']) / 3
    sxx_max_tension = esfuerzos_df.loc[esfuerzos_df['Sxx'].idxmax()]; sxx_max_compresion = esfuerzos_df.loc[esfuerzos_df['Sxx'].idxmin()]
    elem_tens_sxx = elementos.loc[sxx_max_tension.name]; elem_comp_sxx = elementos.loc[sxx_max_compresion.name]
    cx_tens_sxx, cy_tens_sxx = get_centroid(elem_tens_sxx); cx_comp_sxx, cy_comp_sxx = get_centroid(elem_comp_sxx)
    logger.write(f"  - Esfuerzo Horizontal (Sxx):\n"); logger.write(f"    - Máxima Tensión:   {sxx_max_tension['Sxx']/10000:.4f} kg/cm² (en x≈{cx_tens_sxx:.2f}, y≈{cy_tens_sxx:.2f} m)\n"); logger.write(f"    - Máxima Compresión: {sxx_max_compresion['Sxx']/10000:.4f} kg/cm² (en x≈{cx_comp_sxx:.2f}, y≈{cy_comp_sxx:.2f} m)\n\n")
    syy_max_tension = esfuerzos_df.loc[esfuerzos_df['Syy'].idxmax()]; syy_max_compresion = esfuerzos_df.loc[esfuerzos_df['Syy'].idxmin()]
    elem_tens_syy = elementos.loc[syy_max_tension.name]; elem_comp_syy = elementos.loc[syy_max_compresion.name]
    cx_tens_syy, cy_tens_syy = get_centroid(elem_tens_syy); cx_comp_syy, cy_comp_syy = get_centroid(elem_comp_syy)
    logger.write(f"  - Esfuerzo Vertical (Syy):\n"); logger.write(f"    - Máxima Tensión:   {syy_max_tension['Syy']/10000:.4f} kg/cm² (en x≈{cx_tens_syy:.2f}, y≈{cy_tens_syy:.2f} m)\n"); logger.write(f"    - Máxima Compresión: {syy_max_compresion['Syy']/10000:.4f} kg/cm² (en x≈{cx_comp_syy:.2f}, y≈{cy_comp_syy:.2f} m)\n")
    logger.write("="*85 + "\n")
def run_fem_analysis(data, geo_results, method, load_case, logger):
    print("\n" + "#"*85); print(f"  INICIANDO ANÁLISIS FEM: {method.upper()} - CARGAS {'SÍSMICAS' if load_case == 'seismic' else 'ESTÁTICAS'}  "); print("#"*85)
    try:
        mesh_p = float(input("Ingrese el parámetro de densidad de la malla (e.g., 6.0 para una malla fina): "))
        escala_vis = float(input("Ingrese el factor de escala para visualizar la deformada (e.g., 100): "))
    except ValueError: print("Entrada no válida. Usando valores por defecto (malla=6.0, escala=100)."); mesh_p, escala_vis = 6.0, 100.0
    fc = data['fc']; E_mpa = 4700 * math.sqrt(fc); E_kgf_cm2 = E_mpa * 10.1972; E_kgf_m2 = E_kgf_cm2 * 10000; v = 0.2; th = 1.0
    print(f"\n1. Propiedades del material (Concreto):\n   - f'c: {fc:.2f} MPa\n   - Módulo de Elasticidad (E): {E_mpa:.2f} MPa ≈ {E_kgf_cm2:.2f} kgf/cm²")
    print("\n2. Generando malla..."); nudos, elementos, nudos_base, perimetro = fem_generar_malla(data, mesh_p)
    print(f"   ✅ Malla generada con {len(nudos)} nudos y {len(elementos)} elementos.")
    print("\n3. Calculando resortes de la cimentación (Modelo de Winkler)...")
    qu_estatico_kpa = geo_results.get('qu_estatico', 0)
    if qu_estatico_kpa == 0:
        print("   [ADVERTENCIA] Capacidad de carga (qu) es cero. No se pueden calcular los resortes."); return
    s_admisible_m = 0.0254; ks_kn_m3 = qu_estatico_kpa / s_admisible_m; ks_kgf_m3 = ks_kn_m3 * 101.972
    print(f"   - Coeficiente de balasto (ks) calculado: {ks_kn_m3:.2f} kN/m³")
    plt.figure(figsize=(10, 8)); plt.gca().set_aspect('equal', adjustable='box'); plt.triplot(nudos['x'], nudos['y'], elementos[['n1', 'n2', 'n3']].values - 1, 'g-', lw=0.5)
    plt.plot(perimetro[:, 0], perimetro[:, 1], 'b-', lw=2, label='Contorno del Muro'); plt.scatter(nudos_base['x'], nudos_base['y'], c='springgreen', s=80, zorder=5, marker='v', label='Apoyo Elástico (Resortes)')
    anchor_node = nudos.loc[nudos['nudo'] == 1]; plt.scatter(anchor_node['x'], anchor_node['y'], c='red', s=100, zorder=6, marker='^', label='Anclaje Horizontal')
    plt.title('Modelo FEM: Malla con Apoyo Elástico', fontsize=16); plt.xlabel('X (m)'); plt.ylabel('Y (m)'); plt.grid(True); plt.legend(); plt.show()
    print("\n4. Aplicando empujes del terreno..."); fexterna = fem_aplicar_empuje(nudos, data, geo_results, load_case)
    print("\n5. Ensamblando matriz de rigidez y añadiendo resortes..."); NDOF = 2 * len(nudos); K_global = np.zeros((NDOF, NDOF))
    for _, el in elementos.iterrows():
        n_ids = el[['n1', 'n2', 'n3']].astype(int).values; nudos_el_df = nudos[nudos['nudo'].isin(n_ids)].set_index('nudo').loc[n_ids]
        K_local = fem_matriz_rigidez_elemento(E_kgf_m2, v, th, nudos_el_df); dof_map = np.array([[2*(n-1), 2*(n-1)+1] for n in n_ids]).flatten()
        K_global[np.ix_(dof_map, dof_map)] += K_local
    base_nodes_sorted = nudos_base.sort_values('x'); base_x_coords = base_nodes_sorted['x'].values
    for i_num, (idx, row) in enumerate(base_nodes_sorted.iterrows()):
        node_id = int(row['nudo']); x_pos = row['x']
        if i_num == 0: tributary_width = (base_x_coords[1] - x_pos) / 2.0
        elif i_num == len(base_nodes_sorted) - 1: tributary_width = (x_pos - base_x_coords[-2]) / 2.0
        else: tributary_width = (base_x_coords[i_num+1] - base_x_coords[i_num-1]) / 2.0
        spring_stiffness_k = ks_kgf_m3 * tributary_width
        dof_y_index = 2 * (node_id - 1) + 1; K_global[dof_y_index, dof_y_index] += spring_stiffness_k
    F_vector = fexterna.sort_values('nudo')[['fx', 'fy']].to_numpy().flatten()
    gdl_restringidos = []; anchor_node_id = int(nudos_base.loc[nudos_base['x'].idxmin()]['nudo']); dof_x_anchor = 2 * (anchor_node_id - 1); gdl_restringidos.append(dof_x_anchor)
    gdl_libres = sorted(list(set(range(NDOF)) - set(gdl_restringidos)))
    try:
        K_reducida = K_global[np.ix_(gdl_libres, gdl_libres)]; F_reducido = F_vector[gdl_libres]
        desplazamientos_libres = np.linalg.solve(K_reducida, F_reducido)
    except np.linalg.LinAlgError: print("\n❌ ERROR: Matriz de rigidez singular. El modelo es inestable."); return
    desplazamientos = np.zeros(NDOF); desplazamientos[gdl_libres] = desplazamientos_libres; print("   ✅ Sistema resuelto.")
    print("\n6. Post-procesando..."); desplazamientos_df = pd.DataFrame({'nudo': nudos['nudo'], 'dx_mm': desplazamientos[0::2] * 1000, 'dy_mm': desplazamientos[1::2] * 1000})
    esfuerzos_df = fem_visualizar_resultados(nudos, elementos, desplazamientos, E_kgf_m2, v, perimetro, escala_visual=escala_vis)
    export_fem_results_to_file(logger, data, nudos, desplazamientos_df, elementos, esfuerzos_df, method, load_case); print("\n¡Análisis FEM completado! ✅")

# --- BUCLE PRINCIPAL DEL PROGRAMA ---
def main_program():
    current_wall_data = None; rankine_results_cache = None; coulomb_results_cache = None
    log_filename = f"Analisis_Muro_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    while True:
        print("\n" + "="*50); print("--- MENÚ PRINCIPAL: ANÁLISIS DE MUROS ---")
        if current_wall_data: print(">> Datos del muro cargados. Listo para análisis. <<")
        else: print(">> No hay datos cargados. Ingrese datos (Opción 1). <<")
        print("="*50); print("1. Ingresar/Actualizar dimensiones y parámetros"); print("2. Graficar sección transversal del muro"); print("3. Ejecutar Análisis Geotécnico y Diseño de Refuerzo"); print("4. Ejecutar Análisis por Elementos Finitos (FEM)"); print("5. Salir")
        choice = input("Ingrese su opción (1-5): ").strip()
        if choice == '1': current_wall_data = get_input_data(current_wall_data); rankine_results_cache = None; coulomb_results_cache = None
        elif choice == '2':
            if current_wall_data: graficar_muro_contencion(current_wall_data); plt.show()
            else: print("\n❌ No hay datos del muro para graficar. Use la opción 1 primero.")
        elif choice == '3':
            if current_wall_data:
                logger = Logger(log_filename, mode='w')
                rankine_results_cache, coulomb_results_cache = run_both_analyses(current_wall_data, logger)
                logger.close()
                print(f"\n✅ Análisis completo. Resultados guardados en: {log_filename}")
            else: print("\n❌ No hay datos del muro para analizar. Use la opción 1 primero.")
        elif choice == '4':
            if current_wall_data:
                if not rankine_results_cache or not coulomb_results_cache:
                    print("\n⚠️ El análisis FEM requiere los cálculos geotécnicos previos."); print("   Ejecutando análisis silencioso primero...")
                    rankine_results_cache = perform_geotechnical_calculations(current_wall_data, method='rankine')
                    coulomb_results_cache = perform_geotechnical_calculations(current_wall_data, method='coulomb'); print("   ✅ Cálculos completados.")
                print("\n--- Sub-menú: Análisis por Elementos Finitos ---"); print("Seleccione el tipo de carga a aplicar:"); print("1. Cargas Estáticas (Método de Rankine)"); print("2. Cargas Estáticas (Método de Coulomb)"); print("3. Cargas Sísmicas (Método de Rankine)"); print("4. Cargas Sísmicas (Método de Coulomb)")
                fem_choice = input("Ingrese su opción (1-4): ").strip()
                logger_fem = Logger(log_filename, mode='a')
                if fem_choice == '1': run_fem_analysis(current_wall_data, rankine_results_cache, 'Rankine', 'static', logger_fem)
                elif fem_choice == '2': run_fem_analysis(current_wall_data, coulomb_results_cache, 'Coulomb', 'static', logger_fem)
                elif fem_choice == '3': run_fem_analysis(current_wall_data, rankine_results_cache, 'Rankine', 'seismic', logger_fem)
                elif fem_choice == '4': run_fem_analysis(current_wall_data, coulomb_results_cache, 'Coulomb', 'seismic', logger_fem)
                else: print("Opción no válida.")
                logger_fem.close()
                if fem_choice in ['1','2','3','4']: print(f"✅ Resultados del FEM añadidos a: {log_filename}")
            else: print("\n❌ No hay datos del muro para el análisis FEM. Use la opción 1 primero.")
        elif choice == '5': print("\nSaliendo del programa. ¡Hasta luego! 👋"); break
        else: print("\nOpción no válida. Por favor, intente de nuevo.")

if __name__ == "__main__":
    def full_run_both_analyses(data, logger):
        original_stdout = sys.stdout; sys.stdout = logger
        rankine_results = None; coulomb_results = None
        try:
            rankine_results = perform_geotechnical_calculations(data, method='rankine'); print_geotechnical_report(rankine_results, "Rankine")
            coulomb_results = perform_geotechnical_calculations(data, method='coulomb'); print_geotechnical_report(coulomb_results, "Coulomb")
            print("\n\n" + "#"*85); print("          DISEÑO ESTRUCTURAL BASADO EN RESULTADOS DE RANKINE         "); print("#"*85)
            presiones_rankine = {'qpie_est': rankine_results['qpie_estatico'],'qtalon_est': rankine_results['qtalon_estatico'],'qpie_sis': rankine_results.get('qpie_seismic',0),'qtalon_sis': rankine_results.get('qtalon_seismic',0)}
            realizar_diseno_refuerzo(data, rankine_results['Ka'], rankine_results.get('Kae', rankine_results['Ka']), presiones_rankine)
            print("\n\n" + "#"*85); print("                 DISEÑO ESTRUCTURAL BASADO EN RESULTADOS DE COULOMB                "); print("#"*85)
            presiones_coulomb = {'qpie_est': coulomb_results['qpie_estatico'],'qtalon_est': coulomb_results['qtalon_estatico'],'qpie_sis': coulomb_results.get('qpie_seismic',0),'qtalon_sis': coulomb_results.get('qtalon_seismic',0)}
            realizar_diseno_refuerzo(data, coulomb_results['Ka'], coulomb_results.get('Kae', coulomb_results['Ka']), presiones_coulomb)
        finally:
            sys.stdout = original_stdout
            
        if rankine_results and coulomb_results:
            print("\nGenerando diagramas comparativos geotécnicos...")
            try:
                generate_all_diagrams(rankine_results, coulomb_results)
                print("✅ Diagramas geotécnicos generados exitosamente.")
            except Exception as e: print(f"❌ Error al generar los diagramas geotécnicos: {e}")
        return rankine_results, coulomb_results
    run_both_analyses = full_run_both_analyses
    main_program()