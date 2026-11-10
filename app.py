"""
Aplicación Web Interactiva: Simulador de Pirólisis de Biomasa (Quebracho Colorado)
y Diseño Termodinámico Inverso de Sistema de Condensación Pasiva por Tiro Natural.
Flujo Principal: Madera Cargada (kg o m³) + Meta de Litros / % Captura -> Recomendación Constructiva
(Módulos Tipo Tambor 200 L OP-02 y Columna Tubular / Serpentín).
TFI Ingeniería - Máquinas Térmicas.
"""

import importlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import models.pyrolysis_model as _pm
import models.thermal_model as _tm

importlib.reload(_pm)
importlib.reload(_tm)

from models.pyrolysis_model import (
    QUEBRACHO_PROPS,
    calcular_humedad_base_humeda,
    estimar_rendimientos_por_temperatura,
    resolver_diseno_inverso_produccion,
    resolver_balance_experimental,
    generar_curva_temperatura_rendimiento,
    generar_curva_temporal_pirolisis,
)
from models.thermal_model import (
    TUBERIAS_COMERCIALES,
    MATERIALES_TUBO,
    TAMBOR_200L_SPECS,
    disenar_condensador_pasivo,
    recomendar_configuracion_constructiva,
    comparar_diametros_comerciales,
    generar_perfil_axial_temperatura,
)


# =============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS VISUALES
# =============================================================================
st.set_page_config(
    page_title="Simulador Diseño Inverso Pirólisis & Condensador Pasivo | TFI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #4a1c1c 0%, #7c2d12 50%, #1e3a5f 100%);
        padding: 1.2rem 1.6rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        color: #ffffff !important;
        font-size: 1.7rem;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #f3f4f6 !important;
        margin: 0;
        font-size: 0.96rem;
    }
    .status-ok {
        background-color: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        color: #065f46;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .status-warn {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        color: #991b1b;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .rec-card-drum {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 2px solid #f59e0b;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        color: #78350f;
        height: 100%;
    }
    .rec-card-drum h3 {
        color: #92400e !important;
        margin-top: 0;
        font-size: 1.2rem;
    }
    .rec-card-pipe {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 2px solid #3b82f6;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        color: #1e3a8a;
        height: 100%;
    }
    .rec-card-pipe h3 {
        color: #1d4ed8 !important;
        margin-top: 0;
        font-size: 1.2rem;
    }
    .rec-highlight {
        font-size: 1.12rem;
        font-weight: 700;
        margin: 0.6rem 0;
        padding: 0.6rem 0.8rem;
        background-color: rgba(255, 255, 255, 0.75);
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
        <h1>🔥 Simulador de Diseño Inverso: Pirólisis de Quebracho y Condensación Pasiva</h1>
        <p><strong>TFI Ingeniería — Máquinas Térmicas</strong> | Carga de Madera + Meta de Litros / % Captura ➔ Dimensionamiento de Tambores Condensadores (OP-02) y Columnas por Tiro Natural</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# FICHA TÉCNICA DESPLEGABLE DE CÁTEDRA (QUEBRACHO COLORADO)
# =============================================================================
with st.expander("🌲 Ficha Técnica de Referencia de Cátedra: Quebracho Colorado (Schinopsis balansae)", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Composición Elemental (Base Seca)**")
        st.markdown(
            f"- **Carbono (C):** `{QUEBRACHO_PROPS['composicion_elemental_seca_pct']['C']:.1f} %`\n"
            f"- **Hidrógeno (H):** `{QUEBRACHO_PROPS['composicion_elemental_seca_pct']['H']:.1f} %`\n"
            f"- **Oxígeno (O):** `{QUEBRACHO_PROPS['composicion_elemental_seca_pct']['O']:.1f} %`\n"
            f"- **Cenizas:** `{QUEBRACHO_PROPS['composicion_elemental_seca_pct']['Cenizas']:.1f} %`\n"
            f"- **Fórmula Empírica:** `{QUEBRACHO_PROPS['formula_empirica']}`"
        )
    with c2:
        st.markdown("**Poder Calorífico Inferior (Dulong)**")
        st.markdown(
            f"- **$\\text{{PCI}}_{{\\text{{seco}}}}$:** `{QUEBRACHO_PROPS['pci_seco_kcal_kg']:.2f} kcal/kg`\n"
            f"- **$\\text{{PCI}}_{{15\\%\\text{{ H}}}}$:** `{QUEBRACHO_PROPS['pci_15pct_hum_kcal_kg']:.2f} kcal/kg`\n"
            "- **Ecuación Húmeda:** $\\text{PCI}_h = \\text{PCI}_{\\text{seco}} - 600 \\cdot (\\%H/100)$"
        )
    with c3:
        st.markdown("**Densidades y Factor de Estiba**")
        st.markdown(
            f"- **Densidad madera seca ($\\rho$):** `{QUEBRACHO_PROPS['densidad_real_seca_kg_m3']:.0f} kg/m³`\n"
            f"- **Coef. de apilado ($C_a$):** `{QUEBRACHO_PROPS['coeficiente_apilado_Ca']:.2f}`\n"
            f"- **Densidad aparente ($\\rho'$):** `{QUEBRACHO_PROPS['densidad_aparente_estiba_kg_m3']:.0f} kg/m³`\n"
            f"- **Densidad bioinsumo líquido:** `{QUEBRACHO_PROPS['densidad_bioinsumo_kg_l']:.2f} kg/L`"
        )
    with c4:
        st.markdown("**Rendimientos Estequiométricos (Base Seca)**")
        st.markdown(
            f"- **Carbón Sólido:** `{QUEBRACHO_PROPS['rendimientos_teoricos_10h_pct']['carbon_solido']:.2f} %`\n"
            f"- **Ácido Piroleñoso:** `{QUEBRACHO_PROPS['rendimientos_teoricos_10h_pct']['acido_pirolenoso']:.2f} %`\n"
            f"- **Alquitrán Vegetal:** `{QUEBRACHO_PROPS['rendimientos_teoricos_10h_pct']['alquitran_vegetal']:.2f} %`\n"
            f"- **Líquidos Totales ($R_l$):** `{QUEBRACHO_PROPS['rendimientos_teoricos_10h_pct']['liquido_total']:.2f} %`\n"
            f"- **Gases Incondensables ($R_g$):** `{QUEBRACHO_PROPS['rendimientos_teoricos_10h_pct']['gases_incondensables']:.2f} %`"
        )


# =============================================================================
# BARRA LATERAL: PARÁMETROS TÉRMICOS Y CONSTRUCTIVOS
# =============================================================================
st.sidebar.header("⚙️ Parámetros Térmicos del Condensador")

st.sidebar.subheader("1. Temperaturas de Diseño")
temp_horno_c = st.sidebar.slider(
    "Temperatura del Horno de Pirólisis (°C)",
    min_value=350,
    max_value=550,
    value=400,
    step=5,
)
t_entrada_cond_c = st.sidebar.slider(
    "Temperatura Entrada Condensador T_e (°C)",
    min_value=140,
    max_value=190,
    value=160,
    step=5,
    help="Temperatura de los humos al ingresar al condensador / trampa en 'Y' (150-180 °C).",
)
t_saturacion_c = st.sidebar.slider(
    "Temperatura de Saturación / Rocío T_sat (°C)",
    min_value=85,
    max_value=110,
    value=100,
    step=1,
)
t_salida_diseno_c = st.sidebar.slider(
    "Temperatura Salida Target T_s (°C)",
    min_value=30,
    max_value=45,
    value=40,
    step=1,
    help="Punto de rocío target (< 45 °C) para asegurar condensación de piroleñoso y alquitrán.",
)
t_ambiente_c = st.sidebar.slider(
    "Temperatura Ambiente Exterior T_amb (°C)",
    min_value=5,
    max_value=40,
    value=25,
    step=1,
)

st.sidebar.divider()
st.sidebar.subheader("2. Coeficiente de Transferencia U y Material")
material_global = st.sidebar.selectbox(
    "Material Metálico del Condensador",
    options=list(MATERIALES_TUBO.keys()),
    index=2,  # Acero al Carbono / Chapa por defecto para tambores y caños
)
modo_u = st.sidebar.radio(
    "Modelo de Coeficiente Global U",
    options=["Churchill-Chu", "Fijo Cátedra"],
    index=0,
    help="Churchill-Chu evalúa convección natural sobre cilindro vertical; 'Fijo Cátedra' adopta U ≈ 10 W/m²·K.",
)

if modo_u == "Fijo Cátedra":
    u_fijo_val = st.sidebar.number_input(
        "Valor de U fijo (W/m²·K)",
        min_value=2.0,
        max_value=50.0,
        value=10.0,
        step=0.5,
    )
    incluir_rad = False
else:
    u_fijo_val = 10.0
    incluir_rad = st.sidebar.checkbox(
        "Sumar aporte por radiación superficial",
        value=False,
    )

with st.sidebar.expander("🔬 Propiedades Termofísicas y Tambores"):
    cp_gas_in = st.number_input("Cp gases incondensables (kJ/kg·K)", value=1.10, step=0.05)
    cp_vap_in = st.number_input("Cp vapores recalentados (kJ/kg·K)", value=1.95, step=0.05)
    h_fg_in = st.number_input("Calor latente h_fg (kJ/kg)", value=2257.0, step=10.0)
    cp_liq_in = st.number_input("Cp condensado líquido (kJ/kg·K)", value=4.18, step=0.05)
    dens_bio_in = st.number_input("Densidad bioinsumo líquido (kg/L)", value=1.04, step=0.01)
    area_tambor_in = st.number_input(
        "Área efectiva por Tambor de 200 L (m²)",
        value=float(TAMBOR_200L_SPECS["area_efectiva_disipacion_m2"]),
        step=0.05,
        help="Un tambor estándar de 200 L (Ø 0,585 m × 0,88 m) tiene 1,62 m² laterales + tapas ≈ 1,75 m² efectivos.",
    )
    h_max_col_in = st.number_input(
        "Altura máxima práctica por columna tubular (m)",
        min_value=1.5,
        max_value=6.0,
        value=3.0,
        step=0.25,
    )


# =============================================================================
# PESTAÑAS DE LA APLICACIÓN
# =============================================================================
tab_inverso, tab_lab, tab_teoria = st.tabs(
    [
        "🎯 1. Diseño Inverso por Meta de Producción (Madera + Litros ➔ Condensador)",
        "🧪 2. Planilla Post-Ensayo de Laboratorio (Cierre Gravimétrico)",
        "📐 3. Marco Teórico, Ecuaciones y Esquemas Constructivos",
    ]
)


# =============================================================================
# PESTAÑA 1: DISEÑO INVERSO POR META DE PRODUCCIÓN (FLUJO PRINCIPAL)
# =============================================================================
with tab_inverso:
    st.subheader("🎯 Diseño Inverso: De la Carga de Madera y Meta de Litros al Dimensionamiento del Condensador")
    st.markdown(
        "Defina la **carga de madera de Quebracho**, el **tiempo estimado de la quema ($\\Delta t$)** y su **objetivo de producción de bioinsumo** "
        "(en **litros deseados** o en **% de captura $\\eta$**). El simulador verifica la viabilidad estequiométrica, calcula la potencia térmica "
        "y recomienda la configuración constructiva óptima."
    )

    # Presets rápidos opcionales para facilitar pruebas de distintas escalas
    preset_escala = st.segmented_control(
        "Plantillas rápidas de carga (opcional):",
        options=[
            "Horno Media Naranja Monte (12 m³ / 48 h)",
            "Retorta Tambor 200 L (100 kg / 6 h)",
            "Ensayo Piloto (10 kg / 3 h)",
        ],
        default="Horno Media Naranja Monte (12 m³ / 48 h)",
    )

    if preset_escala == "Horno Media Naranja Monte (12 m³ / 48 h)":
        modo_carga_def = "En Metros Cúbicos estibados (m³)"
        val_kg_def = 10800.0
        val_m3_def = 12.0
        t_h_def = 48.0
        litros_def = 400.0
        eta_def = 15.0
    elif preset_escala == "Retorta Tambor 200 L (100 kg / 6 h)":
        modo_carga_def = "En Kilogramos (kg)"
        val_kg_def = 100.0
        val_m3_def = 0.11
        t_h_def = 6.0
        litros_def = 25.0
        eta_def = 50.0
    else:
        modo_carga_def = "En Kilogramos (kg)"
        val_kg_def = 10.0
        val_m3_def = 0.011
        t_h_def = 3.0
        litros_def = 2.5
        eta_def = 50.0

    # -------------------------------------------------------------------------
    # BLOQUE 1: INPUTS OBLIGATORIOS DEL USUARIO
    # -------------------------------------------------------------------------
    col_in1, col_in2, col_in3 = st.columns([1.15, 1.0, 1.25])

    with col_in1:
        st.markdown("#### 1️⃣ Carga de Madera y Humedad")
        tipo_ingreso_madera = st.radio(
            "Modo de ingreso de carga de Quebracho:",
            options=["En Metros Cúbicos estibados (m³)", "En Kilogramos (kg)"],
            index=0 if modo_carga_def.startswith("En Metros") else 1,
            horizontal=True,
        )
        if tipo_ingreso_madera.startswith("En Metros"):
            modo_carga_cod = "m3"
            valor_carga = st.number_input(
                "Volumen de madera cargada (m³)",
                min_value=0.01,
                max_value=500.0,
                value=float(val_m3_def),
                step=0.5,
            )
            coef_apilado_in = st.slider(
                "Coeficiente de apilado / estiba (C_a)",
                min_value=0.50,
                max_value=0.90,
                value=0.75,
                step=0.05,
                help="Con ρ_real = 1200 kg/m³ y C_a = 0,75, la densidad aparente es ρ' = 900 kg/m³.",
            )
            st.caption(
                f"📌 Equivalente a **{valor_carga * 1200.0 * coef_apilado_in:,.1f} kg** de madera "
                f"($\\rho' = {1200.0 * coef_apilado_in:.0f}\\text{{ kg/m}}^3$)."
            )
        else:
            modo_carga_cod = "kg"
            valor_carga = st.number_input(
                "Masa de madera húmeda cargada M (kg)",
                min_value=0.1,
                max_value=200000.0,
                value=float(val_kg_def),
                step=10.0 if val_kg_def >= 50 else 0.5,
            )
            coef_apilado_in = 0.75
            st.caption(
                f"📌 Volumen estibado equivalente: **{valor_carga / 900.0:,.3f} m³** ($\\rho' = 900\\text{{ kg/m}}^3$)."
            )

        humedad_in_pct = st.slider(
            "Humedad de la madera (%H base húmeda)",
            min_value=0.0,
            max_value=45.0,
            value=15.0,
            step=1.0,
        )

    with col_in2:
        st.markdown("#### 2️⃣ Tiempo de Quema / Ciclo ($\\Delta t$)")
        tiempo_quema_h = st.number_input(
            "Duración estimada de la quema Δt (horas)",
            min_value=0.25,
            max_value=240.0,
            value=float(t_h_def),
            step=1.0 if t_h_def >= 6 else 0.5,
            help="Ej.: 1 a 6 h para ensayos en retorta/tambor, o 48 a 72 h para hornos Media Naranja en monte.",
        )
        incluir_agua_hum_in = st.checkbox(
            "Incluir humedad evaporada de la leña en el volumen de condensado",
            value=True,
            help="El agua de humedad libre de la madera se evapora durante la quema y condensa junto con el ácido piroleñoso.",
        )
        usar_curva_temp = st.checkbox(
            f"Ajustar rendimiento teórico según T_horno ({temp_horno_c} °C)",
            value=False,
            help="Si se desmarca, utiliza el rendimiento nominal de cátedra (43,60% líquidos y 39,87% carbón en base seca).",
        )

    with col_in3:
        st.markdown("#### 3️⃣ Objetivo de Producción (Meta)")
        selector_meta = st.radio(
            "Definir meta de condensación por:",
            options=[
                "Opción A: Por Litros deseados a cosechar (L)",
                "Opción B: Por Rendimiento de captura η (%)",
            ],
            index=0,
        )
        if selector_meta.startswith("Opción A"):
            modo_obj_cod = "litros"
            litros_meta_in = st.number_input(
                "Litros de bioinsumo (piroleñoso + alquitrán) deseados por hornada (L)",
                min_value=0.05,
                max_value=100000.0,
                value=float(litros_def),
                step=10.0 if litros_def >= 50 else 0.5,
            )
            eta_meta_in = 15.0
        else:
            modo_obj_cod = "captura_pct"
            eta_meta_in = st.slider(
                "Porcentaje de captura sobre el total de condensables generados η (%)",
                min_value=1.0,
                max_value=100.0,
                value=float(eta_def),
                step=1.0,
            )
            litros_meta_in = 0.0

    # -------------------------------------------------------------------------
    # CÁLCULO INTERNO DEL SIMULADOR (MODELO INVERSO + TÉRMICO)
    # -------------------------------------------------------------------------
    if usar_curva_temp:
        r_calc = estimar_rendimientos_por_temperatura(temp_horno_c)
        r_sol_uso = r_calc["r_solido_pct"]
        r_liq_uso = r_calc["r_liquido_pct"]
    else:
        r_sol_uso = 39.87
        r_liq_uso = 43.60

    res_inv = resolver_diseno_inverso_produccion(
        valor_carga_madera=valor_carga,
        modo_carga_madera=modo_carga_cod,
        humedad_pct=humedad_in_pct,
        tiempo_ciclo_h=tiempo_quema_h,
        modo_objetivo=modo_obj_cod,
        litros_deseados=litros_meta_in,
        eficiencia_captura_pct=eta_meta_in,
        r_solido_pct=r_sol_uso,
        r_liquido_pct=r_liq_uso,
        incluir_humedad_en_bioinsumo=incluir_agua_hum_in,
        densidad_bioinsumo_kg_l=dens_bio_in,
        coef_apilado=coef_apilado_in,
    )

    rec_const = recomendar_configuracion_constructiva(
        m_dot_gas_kg_s=res_inv.m_dot_gas_asociado_kg_s,
        m_dot_cond_kg_s=res_inv.m_dot_cond_objetivo_kg_s,
        litros_objetivo=res_inv.litros_objetivo_efectivo,
        material=material_global,
        t_entrada_c=t_entrada_cond_c,
        t_saturacion_c=t_saturacion_c,
        t_salida_diseno_c=t_salida_diseno_c,
        t_ambiente_c=t_ambiente_c,
        cp_gas_kj_kgk=cp_gas_in,
        cp_vapor_kj_kgk=cp_vap_in,
        h_fg_kj_kg=h_fg_in,
        cp_liquido_kj_kgk=cp_liq_in,
        modo_calculo_u=modo_u,
        u_fijo_w_m2k=u_fijo_val,
        incluir_radiacion_externa=incluir_rad,
        area_efectiva_tambor_m2=area_tambor_in,
        altura_max_columna_practica_m=h_max_col_in,
    )

    st.divider()

    # -------------------------------------------------------------------------
    # BLOQUE 2: DIAGNÓSTICO DE VIABILIDAD ESTEQUIOMÉTRICA Y CAUDALES OBJETIVO
    # -------------------------------------------------------------------------
    if res_inv.saturacion_estequiometrica:
        st.markdown(
            f'<div class="status-warn">{res_inv.mensaje_viabilidad}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="status-ok">{res_inv.mensaje_viabilidad}</div>',
            unsafe_allow_html=True,
        )

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric(
        "Potencial Máximo (100%)",
        f"{res_inv.volumen_cond_total_teor_litros:,.1f} L",
        f"{res_inv.masa_cond_total_teor_kg:,.1f} kg condensables",
    )
    m2.metric(
        "Meta Efectiva de Cosecha",
        f"{res_inv.litros_objetivo_efectivo:,.1f} L",
        f"Captura η = {res_inv.eficiencia_captura_efectiva_pct:.1f} %",
    )
    m3.metric(
        "Caudal de Condensación (ṁ_cond)",
        f"{res_inv.m_dot_cond_objetivo_kg_h:.2f} kg/h",
        f"{res_inv.litros_por_hora_l_h:.2f} L/h ({res_inv.m_dot_cond_objetivo_g_min:.1f} g/min)",
    )
    m4.metric(
        "Potencia Térmica a Disipar (Q̇)",
        f"{rec_const.diseno_optimo_detalle.q_total_w:,.1f} W",
        f"{rec_const.diseno_optimo_detalle.q_total_kcal_h:,.0f} kcal/h",
    )
    m5.metric(
        "Área Externa Requerida (A_ext)",
        f"{rec_const.diseno_optimo_detalle.area_total_requerida_m2:.2f} m²",
        f"U = {rec_const.diseno_optimo_detalle.u_global_w_m2k:.2f} W/m²·K",
    )
    m6.metric(
        "Carbón Sólido en Horno",
        f"{res_inv.masa_carbon_teor_kg:,.1f} kg",
        f"Madera seca: {res_inv.masa_seca_kg:,.1f} kg",
    )

    # -------------------------------------------------------------------------
    # BLOQUE 3: RECOMENDACIÓN CONSTRUCTIVA DIRECTA DEL SOFTWARE (OUTPUT)
    # -------------------------------------------------------------------------
    st.markdown("### 🛠️ Recomendación Constructiva del Sistema de Condensación")

    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.markdown(
            f"""
            <div class="rec-card-drum">
                <h3>🛢️ Opción A: Módulos Tipo Tambor de 200 L (OP-02)</h3>
                <div class="rec-highlight">
                    👉 {rec_const.recomendacion_texto_tambores}
                </div>
                <p><strong>• Cantidad recomendada:</strong> <code>{rec_const.n_tambores_recomendados} módulo(s) de 200 L</code> (cálculo exacto: {rec_const.n_tambores_exacto:.2f} tambores).</p>
                <p><strong>• Disposición constructiva:</strong> {rec_const.disposicion_tambores}</p>
                <p><strong>• Superficie de enfriamiento:</strong> <code>{rec_const.area_instalada_tambores_m2:.2f} m²</code> instalados ({rec_const.area_unitaria_tambor_m2:.2f} m²/tambor) frente a <code>{rec_const.area_requerida_tambores_m2:.2f} m²</code> requeridos (Holgura: +{rec_const.holgura_area_tambores_pct:.1f}%).</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with rec_col2:
        st.markdown(
            f"""
            <div class="rec-card-pipe">
                <h3>🔧 Opción B: Columna Tubular Vertical / Serpentín</h3>
                <div class="rec-highlight">
                    👉 {rec_const.recomendacion_texto_columna}
                </div>
                <p><strong>• Configuración recomendada:</strong> <code>{rec_const.n_columnas_recomendadas} columna(s) de {rec_const.diametro_optimo_recomendado}</code> × <code>{rec_const.altura_por_columna_recomendada_m:.2f} m</code> de altura (Longitud total de caño: <code>{rec_const.longitud_total_cano_m:.2f} m</code>).</p>
                <p><strong>• Verificación de Tiro Natural:</strong> Fuerza motriz <code>ΔP_tiro = {rec_const.delta_p_tiro_optimo_pa:.3f} Pa</code> &gt; Pérdida de carga <code>ΔP_perd = {rec_const.delta_p_perdida_optimo_pa:.3f} Pa</code> (Velocidad: {rec_const.velocidad_optima_m_s:.2f} m/s — <strong>Sin sobrepresión</strong>).</p>
                <p><strong>• Validación Punto de Rocío:</strong> Temperatura de salida estimada <code>T_s = {rec_const.t_salida_optima_c:.1f} °C &lt; 45 °C</code> ✅.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # -------------------------------------------------------------------------
    # BLOQUE 4: AUDITORÍA DETALLADA DE DIÁMETROS, DESGLOSE DE Q Y GRÁFICOS
    # -------------------------------------------------------------------------
    st.markdown("### 📊 Análisis Detallado y Exploración de Diámetros Comerciales")

    exp_c1, exp_c2, exp_c3 = st.columns(3)
    lista_diams = list(TUBERIAS_COMERCIALES.keys())
    with exp_c1:
        diam_manual = st.selectbox(
            "Inspeccionar diámetro comercial específico:",
            options=lista_diams,
            index=lista_diams.index(rec_const.diametro_optimo_recomendado),
        )
    with exp_c2:
        n_tubos_manual = st.number_input(
            "Cantidad de columnas/troneras en paralelo (N_tubos):",
            min_value=1,
            max_value=100,
            value=int(rec_const.n_columnas_recomendadas),
            step=1,
        )
    with exp_c3:
        h_manual = st.number_input(
            "Altura adoptada por columna H (m) [0 = despejar altura exacta L]:",
            min_value=0.0,
            max_value=25.0,
            value=0.0,
            step=0.25,
        )

    diseno_audit = disenar_condensador_pasivo(
        m_dot_gas_kg_s=res_inv.m_dot_gas_asociado_kg_s,
        m_dot_cond_kg_s=res_inv.m_dot_cond_objetivo_kg_s,
        diametro_nominal=diam_manual,
        material=material_global,
        n_tubos=n_tubos_manual,
        t_entrada_c=t_entrada_cond_c,
        t_saturacion_c=t_saturacion_c,
        t_salida_diseno_c=t_salida_diseno_c,
        t_ambiente_c=t_ambiente_c,
        cp_gas_kj_kgk=cp_gas_in,
        cp_vapor_kj_kgk=cp_vap_in,
        h_fg_kj_kg=h_fg_in,
        cp_liquido_kj_kgk=cp_liq_in,
        modo_calculo_u=modo_u,
        u_fijo_w_m2k=u_fijo_val,
        incluir_radiacion_externa=incluir_rad,
        altura_adoptada_m=h_manual,
    )

    df_comp_inv = comparar_diametros_comerciales(
        m_dot_gas_kg_s=res_inv.m_dot_gas_asociado_kg_s,
        m_dot_cond_kg_s=res_inv.m_dot_cond_objetivo_kg_s,
        material=material_global,
        n_tubos=n_tubos_manual,
        t_entrada_c=t_entrada_cond_c,
        t_saturacion_c=t_saturacion_c,
        t_salida_diseno_c=t_salida_diseno_c,
        t_ambiente_c=t_ambiente_c,
        modo_calculo_u=modo_u,
        u_fijo_w_m2k=u_fijo_val,
        incluir_radiacion_externa=incluir_rad,
    )

    st.markdown(f"#### 📏 Tabla Comparativa de Tuberías Comerciales (para {n_tubos_manual} columna/s en paralelo)")
    st.dataframe(df_comp_inv, use_container_width=True, hide_index=True)

    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.markdown(r"#### 🔥 Desglose de Potencia Térmica a Disipar ($\dot{Q}_{total}$)")
        df_q_inv = pd.DataFrame(
            [
                {
                    "Etapa Térmica": "1. Enfriamiento gases incondensables",
                    "Ecuación": "ṁ_gas · Cp_g · (T_e - T_s)",
                    "Potencia (W)": round(diseno_audit.q_gas_sensible_w, 2),
                    "Participación (%)": round(diseno_audit.pct_q_gas, 2),
                },
                {
                    "Etapa Térmica": "2. Desrecalentamiento de vapores",
                    "Ecuación": "ṁ_cond · Cp_v · (T_e - T_sat)",
                    "Potencia (W)": round(diseno_audit.q_vapor_desrecalentamiento_w, 2),
                    "Participación (%)": round(diseno_audit.pct_q_desrec, 2),
                },
                {
                    "Etapa Térmica": "3. Condensación latente (T_sat ≈ 100 °C)",
                    "Ecuación": "ṁ_cond · h_fg (2257 kJ/kg)",
                    "Potencia (W)": round(diseno_audit.q_condensacion_latente_w, 2),
                    "Participación (%)": round(diseno_audit.pct_q_latente, 2),
                },
                {
                    "Etapa Térmica": "4. Subenfriamiento líquido condensado",
                    "Ecuación": "ṁ_cond · Cp_L · (T_sat - T_s)",
                    "Potencia (W)": round(diseno_audit.q_liquido_subenfriamiento_w, 2),
                    "Participación (%)": round(diseno_audit.pct_q_subenf, 2),
                },
            ]
        )
        st.dataframe(df_q_inv, use_container_width=True, hide_index=True)

        fig_dp_inv = go.Figure()
        fig_dp_inv.add_trace(
            go.Bar(
                x=df_comp_inv["Diámetro Nominal"],
                y=df_comp_inv["ΔP Tiro Natural (Pa)"],
                name="Fuerza Motriz ΔP_tiro (Pa)",
                marker_color="#10b981",
            )
        )
        fig_dp_inv.add_trace(
            go.Bar(
                x=df_comp_inv["Diámetro Nominal"],
                y=df_comp_inv["ΔP Pérdidas (Pa)"],
                name="Pérdida de Carga ΔP_perd (Pa)",
                marker_color="#ef4444",
            )
        )
        fig_dp_inv.update_layout(
            title="Efecto Chimenea vs. Pérdida de Carga por Diámetro Comercial",
            xaxis_title="Diámetro Comercial",
            yaxis_title="Presión (Pa)",
            barmode="group",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_dp_inv, use_container_width=True)

    with g_col2:
        st.markdown(f"#### 🌡️ Perfil Axial de Temperatura (Columna de {diam_manual}, H = {diseno_audit.altura_adoptada_m:.2f} m)")
        df_ax_inv = generar_perfil_axial_temperatura(diseno_audit)
        fig_ax_inv = go.Figure()
        fig_ax_inv.add_trace(
            go.Scatter(
                x=df_ax_inv["Altura Axial z (m)"],
                y=df_ax_inv["Temperatura Fluido Interno (°C)"],
                mode="lines",
                name="Gases/Vapores Internos (°C)",
                line=dict(color="#dc2626", width=3),
            )
        )
        fig_ax_inv.add_trace(
            go.Scatter(
                x=df_ax_inv["Altura Axial z (m)"],
                y=df_ax_inv["Temperatura Pared Tubo (°C)"],
                mode="lines",
                name="Pared Metálica (°C)",
                line=dict(color="#f97316", width=2, dash="dot"),
            )
        )
        fig_ax_inv.add_trace(
            go.Scatter(
                x=df_ax_inv["Altura Axial z (m)"],
                y=df_ax_inv["Umbral Máximo Salida (45 °C)"],
                mode="lines",
                name="Target Rocío (45 °C)",
                line=dict(color="#10b981", width=2, dash="dash"),
            )
        )
        fig_ax_inv.update_layout(
            xaxis_title="Altura Axial z (m)",
            yaxis_title="Temperatura (°C)",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_ax_inv, use_container_width=True)


# =============================================================================
# PESTAÑA 2: PLANILLA POST-ENSAYO DE LABORATORIO
# =============================================================================
with tab_lab:
    st.subheader("🧪 Planilla Experimental de Laboratorio (Cierre de Balance Gravimétrico)")
    st.markdown(
        "Permite cargar los datos medidos en laboratorio ($M_{\\text{húmeda}}$, $\\%H$, $M_{\\text{carbón}}$, $M_{\\text{líquido}}$, $\\Delta t$), "
        "incluyendo el **Ensayo de Referencia 2024 ($88{,}5\\text{ g}$ con caño de $3/4''$)**."
    )

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        m_hum_g = st.number_input("Masa húmeda inicial de muestra (g)", min_value=1.0, value=88.5, step=1.0)
        hum_lab = st.number_input("Humedad de la muestra (%H)", min_value=0.0, max_value=60.0, value=0.0, step=0.5)
    with col_p2:
        m_carb_g = st.number_input("Masa de carbón final recolectado (g)", min_value=0.0, value=62.5, step=0.5)
        m_liq_g = st.number_input("Masa de líquido condensado recolectado (g)", min_value=0.0, value=16.7, step=0.5)
    with col_p3:
        t_ensayo_min = st.number_input("Tiempo de ensayo Δt (min)", min_value=1.0, value=60.0, step=5.0)
        diam_ensayo = st.selectbox("Diámetro de caño usado en ensayo", options=lista_diams, index=0)

    bal_lab = resolver_balance_experimental(
        masa_humeda_kg=m_hum_g / 1000.0,
        humedad_pct=hum_lab,
        masa_carbon_kg=m_carb_g / 1000.0,
        masa_liquido_kg=m_liq_g / 1000.0,
        tiempo_pirolisis_min=t_ensayo_min,
    )
    cond_lab = disenar_condensador_pasivo(
        m_dot_gas_kg_s=bal_lab.m_dot_gas_kg_s,
        m_dot_cond_kg_s=bal_lab.m_dot_cond_total_kg_s,
        diametro_nominal=diam_ensayo,
        material=material_global,
        n_tubos=1,
        t_entrada_c=t_entrada_cond_c,
        t_saturacion_c=t_saturacion_c,
        t_salida_diseno_c=t_salida_diseno_c,
        t_ambiente_c=t_ambiente_c,
        altura_adoptada_m=1.0,
    )

    if cond_lab.cumple_tiro_natural:
        st.markdown(f'<div class="status-ok">✅ {cond_lab.diagnostico_global}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-warn">⚠️ {cond_lab.diagnostico_global}</div>', unsafe_allow_html=True)

    l1, l2, l3, l4, l5 = st.columns(5)
    l1.metric("Rendimiento Sólido", f"{bal_lab.rendimiento_solido_pct:.2f} %", f"{m_carb_g:.1f} g carbón")
    l2.metric("Rendimiento Líquido", f"{bal_lab.rendimiento_liquido_pct:.2f} %", f"{m_liq_g:.1f} g líquido")
    l3.metric("Rendimiento Gaseoso", f"{bal_lab.rendimiento_gaseoso_pct:.2f} %", f"{bal_lab.masa_gas_incondensable_kg*1000:.1f} g gas")
    l4.metric("Caudal Másico Gases", f"{bal_lab.m_dot_gas_g_min:.3f} g/min", f"{bal_lab.m_dot_gas_kg_h:.4f} kg/h")
    l5.metric("Potencia Disipada Ensayo", f"{cond_lab.q_total_w:.2f} W", f"Altura req: {cond_lab.longitud_requerida_por_tubo_m:.2f} m")


# =============================================================================
# PESTAÑA 3: MARCO TEÓRICO Y ESQUEMAS CONSTRUCTIVOS
# =============================================================================
with tab_teoria:
    st.subheader("📐 Formulación Matemática del Diseño Inverso y Esquemas de Montaje")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("#### 1. Potencial Estequiométrico y Caudal Objetivo")
        st.latex(r"M_{\text{seca}} = M_{\text{húmeda}} \cdot \left(1 - \frac{\%H}{100}\right)")
        st.latex(r"M_{\text{cond,teor}} = M_{\text{seca}} \cdot R_{\text{líquido}} + M_{\text{agua,humedad}}")
        st.latex(r"\dot{m}_{\text{cond}} = \frac{M_{\text{cond,objetivo}}}{\Delta t} = \frac{V_{\text{objetivo,L}} \cdot \rho_{\text{liq}}}{\Delta t}")
        st.latex(r"\eta_{\text{captura}} = \frac{M_{\text{cond,objetivo}}}{M_{\text{cond,teor}}} \cdot 100 \le 100\%")
    with tc2:
        st.markdown("#### 2. Potencia Térmica, Área Requerida y Tiro Natural")
        st.latex(
            r"\dot{Q}_{\text{total}} = \dot{m}_{\text{gas}} C_{p,g} (T_e - T_s) + "
            r"\dot{m}_{\text{cond}} \left[ C_{p,v} (T_e - T_{\text{sat}}) + h_{fg} + C_{p,L} (T_{\text{sat}} - T_s) \right]"
        )
        st.latex(r"A_{\text{ext,req}} = \frac{\dot{Q}_{\text{total}}}{U \cdot \Delta T_{ml}}")
        st.latex(r"N_{\text{tambores,200L}} = \left\lceil \frac{A_{\text{ext,req}}}{A_{\text{tambor}}} \right\rceil \quad ; \quad L_{\text{caño}} = \frac{A_{\text{ext,req}}}{N_{\text{col}} \cdot \pi \cdot D_{\text{ext}}}")
        st.latex(r"\Delta P_{\text{tiro}} = (\rho_{\text{aire}} - \rho_{\text{gas}}) \cdot g \cdot H > \Delta P_{\text{perd}}")
