# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 20:03:42 2025

@author: USUARIO
"""

# app.py (VERSIÓN FINAL CORREGIDA)
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import sys
from contextlib import redirect_stdout

# Importamos el módulo completo
import muro_analysis_funcs

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Análisis de Muros de Contención",
    page_icon="🧱",
    layout="wide"
)

st.title("🧱 Análisis Geotécnico, Estructural y FEM de Muros de Contención")
st.markdown("Esta aplicación web interactiva implementa los cálculos del script proporcionado, usando Streamlit para la interfaz.")

# --- Barra Lateral para Parámetros de Entrada ---
with st.sidebar:
    st.header("Parámetros de Entrada")
    
    with st.form("input_form"):
        st.subheader("Dimensiones del Muro (m)")
        H2 = st.number_input("Altura pantalla (H2)", value=6.0, step=0.1)
        B = st.number_input("Ancho total zapata (B)", value=4.0, step=0.1)
        H3 = st.number_input("Altura zapata (H3)", value=0.7, step=0.1)
        t_b = st.number_input("Espesor muro base (t_b)", value=0.7, step=0.1)
        t_c = st.number_input("Espesor muro corona (t_c)", value=0.5, step=0.1)
        p_p = st.number_input("Proyección pie zapata (p_p)", value=0.7, step=0.1)
        D = st.number_input("Profundidad cimentación (D)", value=1.5, step=0.1)
        cara_vertical = st.selectbox("Cara vertical del tallo", ('interior', 'exterior'), index=0)

        st.subheader("Parámetros del Suelo y Cargas")
        P1 = st.number_input("Peso esp. relleno (P1 kN/m³)", value=18.0)
        Af1 = st.number_input("Ángulo fricción relleno (Af1 °)", value=30.0)
        C1 = st.number_input("Cohesión relleno (C1 kPa)", value=0.0)
        P2 = st.number_input("Peso esp. cimentación (P2 kN/m³)", value=19.0)
        Af2 = st.number_input("Ángulo fricción cimentación (Af2 °)", value=20.0)
        qsc = st.number_input("Sobrecarga relleno (qsc kPa)", value=10.0)
        Kh = st.number_input("Coef. sísmico horizontal (Kh)", value=0.15, step=0.01)

        st.subheader("Propiedades de Materiales")
        fc = st.number_input("Resistencia concreto f'c (MPa)", value=21.0)
        fy = st.number_input("Fluencia acero fy (MPa)", value=420.0)
        
        submitted = st.form_submit_button("🚀 Analizar Muro")

if submitted:
    data = {
        "H2": H2, "B": B, "H3": H3, "D": D, "t_b": t_b, "t_c": t_c,
        "p_p": p_p, "p_t": B - p_p - t_b, "cara_vertical": cara_vertical,
        "Ai": 10.0, "H1": 0.458, "D2": 0.0, "P1": P1, "Af1": Af1, "C1": C1,
        "P2": P2, "Af2": Af2, "C2": 40.0, "Pem": 23.58, "Pes": 18.0,
        "qsc": qsc, "Kh": Kh, "fc": fc, "fy": fy
    }
    if data["cara_vertical"] == 'interior':
        data["Bb1"], data["Bb3"] = 0.0, data["t_b"] - data["t_c"]
    else:
        data["Bb1"], data["Bb3"] = data["t_b"] - data["t_c"], 0.0
    data["Bb2"] = data["t_c"]
    
    st.session_state.data = data
    st.session_state.analysis_done = True
else:
    st.info("⬅️ Ingrese los parámetros en la barra lateral y presione 'Analizar Muro'.")

if st.session_state.get('analysis_done', False):
    data = st.session_state.data
    
    st.session_state.rankine_results = muro_analysis_funcs.perform_geotechnical_calculations(data, method='rankine')
    st.session_state.coulomb_results = muro_analysis_funcs.perform_geotechnical_calculations(data, method='coulomb')

    tab1, tab2, tab3 = st.tabs(["📊 Resumen y Geotecnia", "🛠️ Diseño de Refuerzo", "💻 Análisis FEM"])

    with tab1:
        st.header("Resumen Gráfico y Geotécnico")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Sección Transversal")
            fig_muro, ax_muro = plt.subplots(figsize=(6, 8))
            muro_analysis_funcs.graficar_muro_contencion(data)
            st.pyplot(fig_muro)
        with col2:
            st.subheader("Verificación de Estabilidad")
            f = io.StringIO()
            with redirect_stdout(f):
                # CORRECCIÓN APLICADA AQUÍ
                muro_analysis_funcs.print_geotechnical_report(st.session_state.rankine_results, "Rankine")
            st.code(f.getvalue(), language='text')
            
            f = io.StringIO()
            with redirect_stdout(f):
                muro_analysis_funcs.print_geotechnical_report(st.session_state.coulomb_results, "Coulomb")
            st.code(f.getvalue(), language='text')

        st.subheader("Diagramas Comparativos")
        with st.spinner("Generando diagramas..."):
            plt.ioff()
            muro_analysis_funcs.generate_all_diagrams(st.session_state.rankine_results, st.session_state.coulomb_results)
            figs = [plt.figure(n) for n in plt.get_fignums()]
            for fig in figs:
                st.pyplot(fig)
            plt.ion()

    with tab2:
        st.header("Diseño de Refuerzo Estructural (E.060)")
        with st.spinner("Calculando acero de refuerzo..."):
            f_refuerzo = io.StringIO()
            with redirect_stdout(f_refuerzo):
                presiones_rankine = {
                    'qpie_est': st.session_state.rankine_results['qpie_estatico'],
                    'qtalon_est': st.session_state.rankine_results['qtalon_estatico'],
                    'qpie_sis': st.session_state.rankine_results.get('qpie_seismic', 0),
                    'qtalon_sis': st.session_state.rankine_results.get('qtalon_seismic', 0)
                }
                muro_analysis_funcs.realizar_diseno_refuerzo(data, st.session_state.rankine_results['Ka'], st.session_state.rankine_results.get('Kae', st.session_state.rankine_results['Ka']), presiones_rankine)
            st.code(f_refuerzo.getvalue(), language='text')
    
    with tab3:
        st.header("Análisis por Elementos Finitos (FEM)")
        st.write("Este análisis utiliza los resultados geotécnicos para aplicar las cargas al modelo FEM.")
        
        col_fem1, col_fem2 = st.columns(2)
        with col_fem1:
            fem_method = st.selectbox("Método de Carga para FEM", ["Rankine", "Coulomb"])
            fem_case = st.selectbox("Caso de Carga para FEM", ["static", "seismic"])
        with col_fem2:
            mesh_p = st.number_input("Densidad de la malla", value=6.0, help="Mayor valor = malla más fina.")
            escala_vis = st.number_input("Factor de escala de la deformada", value=10.0)

        if st.button("Ejecutar Análisis FEM"):
            geo_results_fem = st.session_state.rankine_results if fem_method == "Rankine" else st.session_state.coulomb_results
            with st.spinner("Ejecutando análisis FEM... Esto puede tardar un momento."):
                f_fem = io.StringIO()
                class StreamlitLogger:
                    def write(self, message): f_fem.write(message)
                    def flush(self): pass
                    def close(self): pass

                with redirect_stdout(f_fem):
                    def st_input(prompt):
                        if "malla" in prompt.lower(): return mesh_p
                        if "escala" in prompt.lower(): return escala_vis
                        return ""
                    
                    muro_analysis_funcs.input = st_input
                    muro_analysis_funcs.plt.show = lambda: st.pyplot(plt.gcf())
                    
                    muro_analysis_funcs.run_fem_analysis(data, geo_results_fem, fem_method, fem_case, logger=StreamlitLogger())
                
                st.code(f_fem.getvalue(), language='text')