"""
Módulo de Diseño Termodinámico, Fluidodinámico y Recomendación Constructiva Inversa
del Sistema de Condensación Pasiva por Tiro Natural.
Incluye:
- Balance energético desagregado en 4 etapas (gases, desrecalentamiento, condensación latente, subenfriamiento).
- Correlación de Churchill-Chu para convección natural externa y opción U fijo de cátedra.
- Verificación de Tiro Natural (efecto chimenea) vs. Pérdida de Carga (Darcy-Weisbach + trampa en "Y").
- Recomendación constructiva directa para:
  1) Opción Módulos Tipo Tambor de 200 L (OP-02) conectados a troneras.
  2) Opción Columna Tubular Vertical / Serpentín por diámetro comercial óptimo.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import math
import numpy as np
import pandas as pd


# =============================================================================
# CATÁLOGO DE TUBERÍAS COMERCIALES, MATERIALES Y TAMBOR ESTÁNDAR 200 L (OP-02)
# =============================================================================

TUBERIAS_COMERCIALES: Dict[str, Dict[str, float]] = {
    "3/4\"": {
        "d_nominal_pulg": 0.75,
        "d_ext_mm": 22.22,
        "espesor_mm": 1.24,
        "d_int_mm": 19.74,
    },
    "1\"": {
        "d_nominal_pulg": 1.0,
        "d_ext_mm": 28.58,
        "espesor_mm": 1.27,
        "d_int_mm": 26.04,
    },
    "1,5\"": {
        "d_nominal_pulg": 1.5,
        "d_ext_mm": 41.28,
        "espesor_mm": 1.52,
        "d_int_mm": 38.24,
    },
    "2\"": {
        "d_nominal_pulg": 2.0,
        "d_ext_mm": 53.98,
        "espesor_mm": 1.78,
        "d_int_mm": 50.42,
    },
    "2,5\"": {
        "d_nominal_pulg": 2.5,
        "d_ext_mm": 66.68,
        "espesor_mm": 2.03,
        "d_int_mm": 62.62,
    },
    "3\"": {
        "d_nominal_pulg": 3.0,
        "d_ext_mm": 79.38,
        "espesor_mm": 2.29,
        "d_int_mm": 74.80,
    },
    "4\"": {
        "d_nominal_pulg": 4.0,
        "d_ext_mm": 104.78,
        "espesor_mm": 2.79,
        "d_int_mm": 99.20,
    },
}

MATERIALES_TUBO: Dict[str, Dict[str, float]] = {
    "Cobre (Cu)": {
        "k_w_mk": 385.0,
        "rugosidad_mm": 0.0015,
        "emisividad": 0.65,
    },
    "Aluminio (Al)": {
        "k_w_mk": 205.0,
        "rugosidad_mm": 0.0015,
        "emisividad": 0.45,
    },
    "Acero al Carbono (Chapa / Caño Negro)": {
        "k_w_mk": 50.0,
        "rugosidad_mm": 0.045,
        "emisividad": 0.80,
    },
    "Acero Inoxidable AISI 304": {
        "k_w_mk": 16.2,
        "rugosidad_mm": 0.015,
        "emisividad": 0.60,
    },
}

# Especificaciones geométricas de un Tambor Metálico Estándar de 200 L (Módulo OP-02)
TAMBOR_200L_SPECS = {
    "volumen_nominal_l": 200.0,
    "diametro_m": 0.585,
    "altura_m": 0.880,
    "espesor_chapa_mm": 1.25,
    "area_lateral_m2": math.pi * 0.585 * 0.880,  # ~1,617 m²
    "area_efectiva_disipacion_m2": 1.75,         # Lateral + tapa superior expuesta al aire
}


@dataclass
class CondenserDesignResult:
    """Resultados integrales del diseño térmico y fluidodinámico del condensador pasivo."""
    diametro_nominal: str
    material: str
    n_tubos: int
    d_ext_m: float
    d_int_m: float
    espesor_m: float

    t_entrada_c: float
    t_saturacion_c: float
    t_salida_diseno_c: float
    t_ambiente_c: float
    delta_t_ml_c: float

    q_gas_sensible_w: float
    q_vapor_desrecalentamiento_w: float
    q_condensacion_latente_w: float
    q_liquido_subenfriamiento_w: float
    q_total_w: float
    q_total_kj_h: float
    q_total_kcal_h: float

    pct_q_gas: float
    pct_q_desrec: float
    pct_q_latente: float
    pct_q_subenf: float

    h_ext_conv_w_m2k: float
    h_ext_rad_w_m2k: float
    h_int_efectivo_w_m2k: float
    u_global_w_m2k: float
    rayleigh_L: float
    nusselt_L: float

    area_total_requerida_m2: float
    area_por_tubo_requerida_m2: float
    longitud_requerida_por_tubo_m: float
    altura_adoptada_m: float
    area_adoptada_total_m2: float
    t_salida_estimada_con_altura_adoptada_c: float
    cumple_temp_salida_45c: bool

    rho_aire_amb_kg_m3: float
    rho_mezcla_media_kg_m3: float
    caudal_volumetrico_total_m3_h: float
    caudal_volumetrico_total_l_min: float
    velocidad_interior_m_s: float
    reynolds_interno: float
    regimen_flujo: str
    factor_friccion_darcy: float
    delta_p_tiro_pa: float
    delta_p_friccion_pa: float
    delta_p_accesorios_pa: float
    delta_p_perdida_total_pa: float
    margen_tiro_neto_pa: float
    ratio_tiro_perdida: float
    cumple_tiro_natural: bool
    diagnostico_global: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConstructiveRecommendationResult:
    """
    Recomendación constructiva directa del software para alcanzar la meta de litros/captura:
    1) Opción Módulos Tipo Tambor de 200 L (OP-02)
    2) Opción Columna Tubular / Serpentín por Tiro Natural
    """
    # Opción Módulos Tipo Tambor 200 L (OP-02)
    area_requerida_tambores_m2: float
    u_tambor_w_m2k: float
    area_unitaria_tambor_m2: float
    n_tambores_exacto: float
    n_tambores_recomendados: int
    area_instalada_tambores_m2: float
    holgura_area_tambores_pct: float
    disposicion_tambores: str
    recomendacion_texto_tambores: str

    # Opción Columna Tubular / Serpentín Recomendada Automáticamente
    diametro_optimo_recomendado: str
    n_columnas_recomendadas: int
    altura_por_columna_recomendada_m: float
    longitud_total_cano_m: float
    area_instalada_columnas_m2: float
    delta_p_tiro_optimo_pa: float
    delta_p_perdida_optimo_pa: float
    velocidad_optima_m_s: float
    t_salida_optima_c: float
    recomendacion_texto_columna: str
    diseno_optimo_detalle: CondenserDesignResult

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def propiedades_aire_film(temp_film_c: float) -> Dict[str, float]:
    """Calcula las propiedades termofísicas del aire atmosférico a la temperatura de película."""
    T_k = temp_film_c + 273.15
    p_atm = 101325.0
    r_aire = 287.058

    rho = p_atm / (r_aire * T_k)
    mu = 1.458e-6 * (T_k ** 1.5) / (T_k + 110.4)
    nu = mu / rho
    k = 0.0241 * ((T_k / 273.15) ** 0.82)
    cp = 1006.0
    alpha = k / (rho * cp)
    pr = nu / alpha
    beta = 1.0 / T_k

    return {
        "T_k": T_k,
        "rho": rho,
        "mu": mu,
        "nu": nu,
        "k": k,
        "cp": cp,
        "alpha": alpha,
        "Pr": pr,
        "beta": beta,
    }


def calcular_churchill_chu_vertical(
    longitud_m: float,
    diametro_ext_m: float,
    temp_pared_c: float,
    temp_amb_c: float,
) -> Dict[str, float]:
    """
    Correlación de Churchill y Chu (1975) para convección natural sobre cilindro vertical.
    """
    L = max(longitud_m, 0.15)
    delta_t = max(abs(temp_pared_c - temp_amb_c), 1.0)
    t_film = 0.5 * (temp_pared_c + temp_amb_c)

    props = propiedades_aire_film(t_film)
    g = 9.80665

    gr_L = (g * props["beta"] * delta_t * (L ** 3)) / (props["nu"] ** 2)
    ra_L = gr_L * props["Pr"]

    denom = (1.0 + (0.492 / props["Pr"]) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    nu_placa = (0.825 + (0.387 * (ra_L ** (1.0 / 6.0))) / denom) ** 2

    xi = ((32.0 ** 0.5) / (gr_L ** 0.25)) * (L / max(diametro_ext_m, 0.005))
    factor_curvatura = 1.0 + 0.30 * (xi ** 0.9) if xi > 0.1 else 1.0

    nu_cil = nu_placa * factor_curvatura
    h_conv = (nu_cil * props["k"]) / L

    return {
        "Gr_L": gr_L,
        "Ra_L": ra_L,
        "Pr": props["Pr"],
        "Nu_L": nu_cil,
        "h_conv_w_m2k": h_conv,
    }


def calcular_delta_t_ml(t_entrada_c: float, t_salida_c: float, t_amb_c: float) -> float:
    """Calcula la Diferencia de Temperatura Media Logarítmica (LMTD / ΔT_ml)."""
    dt1 = max(t_entrada_c - t_amb_c, 0.5)
    dt2 = max(t_salida_c - t_amb_c, 0.5)
    if abs(dt1 - dt2) < 1e-4:
        return dt1
    return (dt1 - dt2) / math.log(dt1 / dt2)


def disenar_condensador_pasivo(
    m_dot_gas_kg_s: float,
    m_dot_cond_kg_s: float,
    diametro_nominal: str = "2\"",
    material: str = "Cobre (Cu)",
    n_tubos: int = 1,
    t_entrada_c: float = 160.0,
    t_saturacion_c: float = 100.0,
    t_salida_diseno_c: float = 40.0,
    t_ambiente_c: float = 25.0,
    cp_gas_kj_kgk: float = 1.10,
    cp_vapor_kj_kgk: float = 1.95,
    h_fg_kj_kg: float = 2257.0,
    cp_liquido_kj_kgk: float = 4.18,
    modo_calculo_u: str = "Churchill-Chu",
    u_fijo_w_m2k: float = 10.0,
    incluir_radiacion_externa: bool = False,
    altura_adoptada_m: float = 0.0,
    k_accesorios: float = 2.2,
) -> CondenserDesignResult:
    """
    Ejecuta el dimensionamiento térmico y la verificación de tiro natural del
    condensador tubular vertical.
    """
    if diametro_nominal not in TUBERIAS_COMERCIALES:
        raise ValueError(f"Diámetro '{diametro_nominal}' no encontrado en catálogo comercial.")
    if material not in MATERIALES_TUBO:
        material = list(MATERIALES_TUBO.keys())[0]

    n_tubos = max(int(n_tubos), 1)
    tubo = TUBERIAS_COMERCIALES[diametro_nominal]
    mat = MATERIALES_TUBO[material]

    d_ext_m = tubo["d_ext_mm"] / 1000.0
    d_int_m = tubo["d_int_mm"] / 1000.0
    espesor_m = tubo["espesor_mm"] / 1000.0

    # 1. BALANCE TÉRMICO DESAGREGADO (W = J/s)
    t_sat_ef = min(max(t_saturacion_c, t_salida_diseno_c), t_entrada_c)

    q_gas_w = m_dot_gas_kg_s * (cp_gas_kj_kgk * 1000.0) * max(t_entrada_c - t_salida_diseno_c, 0.0)
    q_desrec_w = m_dot_cond_kg_s * (cp_vapor_kj_kgk * 1000.0) * max(t_entrada_c - t_sat_ef, 0.0)
    q_latente_w = m_dot_cond_kg_s * (h_fg_kj_kg * 1000.0)
    q_subenf_w = m_dot_cond_kg_s * (cp_liquido_kj_kgk * 1000.0) * max(t_sat_ef - t_salida_diseno_c, 0.0)

    q_total_w = q_gas_w + q_desrec_w + q_latente_w + q_subenf_w
    q_total_kj_h = q_total_w * 3.6
    q_total_kcal_h = q_total_kj_h / 4.184

    q_denom = max(q_total_w, 1e-9)
    pct_q_gas = (q_gas_w / q_denom) * 100.0
    pct_q_desrec = (q_desrec_w / q_denom) * 100.0
    pct_q_lat = (q_latente_w / q_denom) * 100.0
    pct_q_sub = (q_subenf_w / q_denom) * 100.0

    # 2. TRANSFERENCIA DE CALOR Y DESPEJE DE ÁREA Y LONGITUD (L)
    delta_t_ml = calcular_delta_t_ml(t_entrada_c, t_salida_diseno_c, t_ambiente_c)
    t_pared_media_c = t_ambiente_c + 0.85 * delta_t_ml

    h_int_w_m2k = 180.0
    r_pared = (d_ext_m * math.log(d_ext_m / d_int_m)) / (2.0 * mat["k_w_mk"])
    r_int = (d_ext_m / d_int_m) * (1.0 / h_int_w_m2k)

    L_est = 1.5
    ra_L = 0.0
    nu_L = 0.0
    h_conv_ext = u_fijo_w_m2k
    h_rad_ext = 0.0

    if modo_calculo_u == "Churchill-Chu":
        for _ in range(8):
            cc = calcular_churchill_chu_vertical(
                longitud_m=L_est,
                diametro_ext_m=d_ext_m,
                temp_pared_c=t_pared_media_c,
                temp_amb_c=t_ambiente_c,
            )
            h_conv_ext = cc["h_conv_w_m2k"]
            ra_L = cc["Ra_L"]
            nu_L = cc["Nu_L"]

            if incluir_radiacion_externa:
                sigma = 5.670374419e-8
                t_p_k = t_pared_media_c + 273.15
                t_a_k = t_ambiente_c + 273.15
                h_rad_ext = (
                    mat["emisividad"]
                    * sigma
                    * (t_p_k ** 2 + t_a_k ** 2)
                    * (t_p_k + t_a_k)
                )
            else:
                h_rad_ext = 0.0

            h_ext_total = h_conv_ext + h_rad_ext
            u_global = 1.0 / ((1.0 / max(h_ext_total, 0.1)) + r_pared + r_int)
            a_req = q_total_w / max(u_global * delta_t_ml, 1e-6)
            L_nuevo = a_req / (n_tubos * math.pi * d_ext_m)
            L_est = max(0.2, 0.5 * L_est + 0.5 * L_nuevo)
    else:
        cc = calcular_churchill_chu_vertical(
            longitud_m=1.5,
            diametro_ext_m=d_ext_m,
            temp_pared_c=t_pared_media_c,
            temp_amb_c=t_ambiente_c,
        )
        ra_L = cc["Ra_L"]
        nu_L = cc["Nu_L"]
        h_conv_ext = u_fijo_w_m2k
        h_rad_ext = 0.0
        u_global = u_fijo_w_m2k

    area_total_req_m2 = q_total_w / max(u_global * delta_t_ml, 1e-6)
    area_por_tubo_req_m2 = area_total_req_m2 / n_tubos
    longitud_req_m = area_por_tubo_req_m2 / (math.pi * d_ext_m)

    h_adoptada_m = altura_adoptada_m if altura_adoptada_m > 0 else longitud_req_m
    area_adoptada_total_m2 = n_tubos * math.pi * d_ext_m * h_adoptada_m

    c_dot_efectivo_w_k = q_total_w / max(t_entrada_c - t_salida_diseno_c, 1.0)
    ntu = (u_global * area_adoptada_total_m2) / max(c_dot_efectivo_w_k, 1e-9)
    t_salida_estimada_c = t_ambiente_c + (t_entrada_c - t_ambiente_c) * math.exp(-ntu)
    cumple_temp_45c = t_salida_estimada_c < 45.0

    # 3. FLUIDODINÁMICA Y VERIFICACIÓN DE TIRO NATURAL (EFECTO CHIMENEA)
    g = 9.80665
    p_atm = 101325.0
    r_univ = 8314.46

    t_amb_k = t_ambiente_c + 273.15
    rho_aire_amb = p_atm / (287.058 * t_amb_k)

    t_media_gas_c = 0.5 * (t_entrada_c + t_salida_estimada_c)
    t_media_gas_k = t_media_gas_c + 273.15

    frac_masa_vapor_media = (0.5 * m_dot_cond_kg_s) / max(m_dot_gas_kg_s + 0.5 * m_dot_cond_kg_s, 1e-9)
    m_molar_mezcla = 1.0 / ((frac_masa_vapor_media / 18.015) + ((1.0 - frac_masa_vapor_media) / 27.5))
    rho_mezcla_media = (p_atm * m_molar_mezcla) / (r_univ * t_media_gas_k)

    caudal_flujo_medio_kg_s = m_dot_gas_kg_s + 0.5 * m_dot_cond_kg_s
    q_vol_total_m3_s = caudal_flujo_medio_kg_s / max(rho_mezcla_media, 1e-5)
    q_vol_total_m3_h = q_vol_total_m3_s * 3600.0
    q_vol_total_l_min = q_vol_total_m3_s * 60000.0

    area_transversal_tubo_m2 = math.pi * (d_int_m ** 2) / 4.0
    area_transversal_total_m2 = n_tubos * area_transversal_tubo_m2
    velocidad_m_s = q_vol_total_m3_s / max(area_transversal_total_m2, 1e-9)

    mu_gas = 1.458e-6 * (t_media_gas_k ** 1.5) / (t_media_gas_k + 110.4)
    reynolds = (rho_mezcla_media * velocidad_m_s * d_int_m) / max(mu_gas, 1e-9)

    if reynolds < 1e-3:
        regimen = "Estanco"
        f_darcy = 0.0
    elif reynolds < 2300.0:
        regimen = "Laminar"
        f_darcy = 64.0 / reynolds
    else:
        regimen = "Turbulento / Transición"
        rug_rel = (mat["rugosidad_mm"] / 1000.0) / d_int_m
        f_darcy = 0.25 / (
            math.log10((rug_rel / 3.7) + (5.74 / (reynolds ** 0.9))) ** 2
        )

    delta_p_tiro_pa = max(rho_aire_amb - rho_mezcla_media, 0.0) * g * h_adoptada_m

    presion_dinamica_pa = 0.5 * rho_mezcla_media * (velocidad_m_s ** 2)
    delta_p_friccion_pa = f_darcy * (h_adoptada_m / d_int_m) * presion_dinamica_pa
    delta_p_accesorios_pa = k_accesorios * presion_dinamica_pa
    delta_p_perdida_total_pa = delta_p_friccion_pa + delta_p_accesorios_pa

    if d_int_m < 0.022 and velocidad_m_s > 0.08:
        penalidad_capilar_pa = 1.8 * (0.022 / d_int_m) ** 2 * delta_p_perdida_total_pa
        delta_p_accesorios_pa += penalidad_capilar_pa
        delta_p_perdida_total_pa = delta_p_friccion_pa + delta_p_accesorios_pa

    margen_tiro_pa = delta_p_tiro_pa - delta_p_perdida_total_pa
    ratio_tiro = delta_p_tiro_pa / max(delta_p_perdida_total_pa, 1e-6)

    cumple_tiro = (margen_tiro_pa > 0.0) and (velocidad_m_s <= 3.5) and (d_int_m >= 0.024)

    if cumple_temp_45c and cumple_tiro:
        diagnostico = "DISEÑO APROBADO: Cumple enfriamiento (< 45 °C) y tiro natural sin sobrepresión."
    elif not cumple_tiro and d_int_m < 0.024:
        diagnostico = (
            "RIESGO DE SOBREPRESIÓN / AHOGAMIENTO: Diámetro insuficiente (< 1\"). "
            "Estrangulamiento bifásico en la trampa en 'Y' (caso reportado en ensayo 2024 con 3/4\")."
        )
    elif not cumple_tiro:
        diagnostico = (
            "NO CUMPLE TIRO NATURAL: La pérdida de carga supera al tiro o la velocidad "
            "genera arrastre. Aumente el diámetro comercial o el número de tubos en paralelo."
        )
    else:
        diagnostico = (
            f"ALERTA TÉRMICA: La temperatura de salida estimada ({t_salida_estimada_c:.1f} °C) "
            "supera el umbral de 45 °C. Incremente la altura de columna o el número de tubos."
        )

    return CondenserDesignResult(
        diametro_nominal=diametro_nominal,
        material=material,
        n_tubos=n_tubos,
        d_ext_m=d_ext_m,
        d_int_m=d_int_m,
        espesor_m=espesor_m,
        t_entrada_c=t_entrada_c,
        t_saturacion_c=t_sat_ef,
        t_salida_diseno_c=t_salida_diseno_c,
        t_ambiente_c=t_ambiente_c,
        delta_t_ml_c=delta_t_ml,
        q_gas_sensible_w=q_gas_w,
        q_vapor_desrecalentamiento_w=q_desrec_w,
        q_condensacion_latente_w=q_latente_w,
        q_liquido_subenfriamiento_w=q_subenf_w,
        q_total_w=q_total_w,
        q_total_kj_h=q_total_kj_h,
        q_total_kcal_h=q_total_kcal_h,
        pct_q_gas=pct_q_gas,
        pct_q_desrec=pct_q_desrec,
        pct_q_latente=pct_q_lat,
        pct_q_subenf=pct_q_sub,
        h_ext_conv_w_m2k=h_conv_ext,
        h_ext_rad_w_m2k=h_rad_ext,
        h_int_efectivo_w_m2k=h_int_w_m2k,
        u_global_w_m2k=u_global,
        rayleigh_L=ra_L,
        nusselt_L=nu_L,
        area_total_requerida_m2=area_total_req_m2,
        area_por_tubo_requerida_m2=area_por_tubo_req_m2,
        longitud_requerida_por_tubo_m=longitud_req_m,
        altura_adoptada_m=h_adoptada_m,
        area_adoptada_total_m2=area_adoptada_total_m2,
        t_salida_estimada_con_altura_adoptada_c=t_salida_estimada_c,
        cumple_temp_salida_45c=cumple_temp_45c,
        rho_aire_amb_kg_m3=rho_aire_amb,
        rho_mezcla_media_kg_m3=rho_mezcla_media,
        caudal_volumetrico_total_m3_h=q_vol_total_m3_h,
        caudal_volumetrico_total_l_min=q_vol_total_l_min,
        velocidad_interior_m_s=velocidad_m_s,
        reynolds_interno=reynolds,
        regimen_flujo=regimen,
        factor_friccion_darcy=f_darcy,
        delta_p_tiro_pa=delta_p_tiro_pa,
        delta_p_friccion_pa=delta_p_friccion_pa,
        delta_p_accesorios_pa=delta_p_accesorios_pa,
        delta_p_perdida_total_pa=delta_p_perdida_total_pa,
        margen_tiro_neto_pa=margen_tiro_pa,
        ratio_tiro_perdida=ratio_tiro,
        cumple_tiro_natural=cumple_tiro,
        diagnostico_global=diagnostico,
    )


def recomendar_configuracion_constructiva(
    m_dot_gas_kg_s: float,
    m_dot_cond_kg_s: float,
    litros_objetivo: float,
    material: str = "Acero al Carbono (Chapa / Caño Negro)",
    t_entrada_c: float = 160.0,
    t_saturacion_c: float = 100.0,
    t_salida_diseno_c: float = 40.0,
    t_ambiente_c: float = 25.0,
    cp_gas_kj_kgk: float = 1.10,
    cp_vapor_kj_kgk: float = 1.95,
    h_fg_kj_kg: float = 2257.0,
    cp_liquido_kj_kgk: float = 4.18,
    modo_calculo_u: str = "Churchill-Chu",
    u_fijo_w_m2k: float = 10.0,
    incluir_radiacion_externa: bool = False,
    area_efectiva_tambor_m2: float = 1.75,
    altura_max_columna_practica_m: float = 3.0,
) -> ConstructiveRecommendationResult:
    """
    A partir de los caudales requeridos por la meta de condensación, genera las dos
    recomendaciones constructivas de ingeniería:
    1) Opción Módulos Tipo Tambor de 200 L (OP-02).
    2) Opción Columna Tubular / Serpentín (seleccionando diámetro comercial y altura).
    """
    # -------------------------------------------------------------------------
    # A. OPCIÓN 1: MÓDULOS TIPO TAMBOR DE 200 L (OP-02)
    # -------------------------------------------------------------------------
    delta_t_ml = calcular_delta_t_ml(t_entrada_c, t_salida_diseno_c, t_ambiente_c)
    t_pared_tambor = t_ambiente_c + 0.85 * delta_t_ml

    if modo_calculo_u == "Churchill-Chu":
        cc_tambor = calcular_churchill_chu_vertical(
            longitud_m=TAMBOR_200L_SPECS["altura_m"],
            diametro_ext_m=TAMBOR_200L_SPECS["diametro_m"],
            temp_pared_c=t_pared_tambor,
            temp_amb_c=t_ambiente_c,
        )
        u_tambor = cc_tambor["h_conv_w_m2k"]
        if incluir_radiacion_externa:
            sigma = 5.670374419e-8
            t_p_k = t_pared_tambor + 273.15
            t_a_k = t_ambiente_c + 273.15
            u_tambor += 0.80 * sigma * (t_p_k ** 2 + t_a_k ** 2) * (t_p_k + t_a_k)
    else:
        u_tambor = u_fijo_w_m2k

    # Evaluamos la potencia térmica con una llamada base
    diseno_base = disenar_condensador_pasivo(
        m_dot_gas_kg_s=m_dot_gas_kg_s,
        m_dot_cond_kg_s=m_dot_cond_kg_s,
        diametro_nominal="2\"",
        material=material,
        n_tubos=1,
        t_entrada_c=t_entrada_c,
        t_saturacion_c=t_saturacion_c,
        t_salida_diseno_c=t_salida_diseno_c,
        t_ambiente_c=t_ambiente_c,
        cp_gas_kj_kgk=cp_gas_kj_kgk,
        cp_vapor_kj_kgk=cp_vapor_kj_kgk,
        h_fg_kj_kg=h_fg_kj_kg,
        cp_liquido_kj_kgk=cp_liquido_kj_kgk,
        modo_calculo_u=modo_calculo_u,
        u_fijo_w_m2k=u_fijo_w_m2k,
        incluir_radiacion_externa=incluir_radiacion_externa,
    )

    q_total_w = diseno_base.q_total_w
    area_req_tambores_m2 = q_total_w / max(u_tambor * delta_t_ml, 1e-6)
    n_tambores_exacto = area_req_tambores_m2 / max(area_efectiva_tambor_m2, 0.1)
    n_tambores_rec = max(1, int(math.ceil(n_tambores_exacto)))
    area_inst_tambores = n_tambores_rec * area_efectiva_tambor_m2
    holgura_tambores_pct = ((area_inst_tambores - area_req_tambores_m2) / max(area_req_tambores_m2, 1e-6)) * 100.0

    if n_tambores_rec == 1:
        disp_tambores = "1 tambor condensador de 200 L conectado directamente a la salida/tronera con trampa inferior."
    elif n_tambores_rec <= 3:
        disp_tambores = (
            f"{n_tambores_rec} tambores de 200 L (en paralelo sobre {n_tambores_rec} troneras del horno "
            f"o en serie tipo tándem con drenaje individual)."
        )
    else:
        disp_tambores = (
            f"{n_tambores_rec} tambores de 200 L distribuidos en paralelo en las troneras periféricas del horno "
            f"(ej. {math.ceil(n_tambores_rec / 2)} módulos simples o {math.ceil(n_tambores_rec / 2)} pares en serie)."
        )

    texto_tambores = (
        f"Para recuperar tus {litros_objetivo:,.1f} L necesitás {n_tambores_rec} "
        f"{'tambor estándar de 200 L' if n_tambores_rec == 1 else 'tambores estándar de 200 L'} "
        f"conectados a las troneras (Área requerida: {area_req_tambores_m2:.2f} m² | "
        f"Área provista: {area_inst_tambores:.2f} m²)."
    )

    # -------------------------------------------------------------------------
    # B. OPCIÓN 2: COLUMNA TUBULAR / SERPENTÍN (SELECCIÓN ÓPTIMA)
    # -------------------------------------------------------------------------
    # Buscamos entre los diámetros comerciales (>= 1") el que permita menor número de tubos
    # con una altura práctica constructiva (entre 0,8 m y altura_max_columna_practica_m) y tiro natural aprobado
    diametros_candidatos = ["1\"", "1,5\"", "2\"", "2,5\"", "3\"", "4\""]
    mejor_diseno = None

    for n_col in [1, 2, 3, 4, 6, 8, 12, 16, 24]:
        for diam in diametros_candidatos:
            cand = disenar_condensador_pasivo(
                m_dot_gas_kg_s=m_dot_gas_kg_s,
                m_dot_cond_kg_s=m_dot_cond_kg_s,
                diametro_nominal=diam,
                material=material,
                n_tubos=n_col,
                t_entrada_c=t_entrada_c,
                t_saturacion_c=t_saturacion_c,
                t_salida_diseno_c=t_salida_diseno_c,
                t_ambiente_c=t_ambiente_c,
                cp_gas_kj_kgk=cp_gas_kj_kgk,
                cp_vapor_kj_kgk=cp_vapor_kj_kgk,
                h_fg_kj_kg=h_fg_kj_kg,
                cp_liquido_kj_kgk=cp_liquido_kj_kgk,
                modo_calculo_u=modo_calculo_u,
                u_fijo_w_m2k=u_fijo_w_m2k,
                incluir_radiacion_externa=incluir_radiacion_externa,
            )
            # Si la longitud requerida es muy corta (< 0,8 m), adoptamos 1,0 m mínimo para buen efecto chimenea
            h_eval = max(cand.longitud_requerida_por_tubo_m, 0.85)
            cand_eval = disenar_condensador_pasivo(
                m_dot_gas_kg_s=m_dot_gas_kg_s,
                m_dot_cond_kg_s=m_dot_cond_kg_s,
                diametro_nominal=diam,
                material=material,
                n_tubos=n_col,
                t_entrada_c=t_entrada_c,
                t_saturacion_c=t_saturacion_c,
                t_salida_diseno_c=t_salida_diseno_c,
                t_ambiente_c=t_ambiente_c,
                cp_gas_kj_kgk=cp_gas_kj_kgk,
                cp_vapor_kj_kgk=cp_vapor_kj_kgk,
                h_fg_kj_kg=h_fg_kj_kg,
                cp_liquido_kj_kgk=cp_liquido_kj_kgk,
                modo_calculo_u=modo_calculo_u,
                u_fijo_w_m2k=u_fijo_w_m2k,
                incluir_radiacion_externa=incluir_radiacion_externa,
                altura_adoptada_m=h_eval,
            )
            if (
                cand_eval.cumple_tiro_natural
                and cand_eval.cumple_temp_salida_45c
                and cand_eval.altura_adoptada_m <= altura_max_columna_practica_m
            ):
                mejor_diseno = cand_eval
                break
        if mejor_diseno is not None:
            break

    # Fallback si la potencia es muy grande para los topes anteriores
    if mejor_diseno is None:
        n_col_necesarias = max(
            1,
            int(
                math.ceil(
                    diseno_base.area_total_requerida_m2
                    / (math.pi * (TUBERIAS_COMERCIALES["3\""]["d_ext_mm"] / 1000.0) * altura_max_columna_practica_m)
                )
            ),
        )
        mejor_diseno = disenar_condensador_pasivo(
            m_dot_gas_kg_s=m_dot_gas_kg_s,
            m_dot_cond_kg_s=m_dot_cond_kg_s,
            diametro_nominal="3\"",
            material=material,
            n_tubos=n_col_necesarias,
            t_entrada_c=t_entrada_c,
            t_saturacion_c=t_saturacion_c,
            t_salida_diseno_c=t_salida_diseno_c,
            t_ambiente_c=t_ambiente_c,
            cp_gas_kj_kgk=cp_gas_kj_kgk,
            cp_vapor_kj_kgk=cp_vapor_kj_kgk,
            h_fg_kj_kg=h_fg_kj_kg,
            cp_liquido_kj_kgk=cp_liquido_kj_kgk,
            modo_calculo_u=modo_calculo_u,
            u_fijo_w_m2k=u_fijo_w_m2k,
            incluir_radiacion_externa=incluir_radiacion_externa,
        )

    long_total_m = mejor_diseno.n_tubos * mejor_diseno.altura_adoptada_m
    if mejor_diseno.n_tubos == 1:
        texto_columna = (
            f"Tubería comercial de {mejor_diseno.diametro_nominal} con altura neta de "
            f"{mejor_diseno.altura_adoptada_m:.2f} metros (1 columna vertical con trampa en 'Y', "
            f"ΔP_tiro = {mejor_diseno.delta_p_tiro_pa:.2f} Pa > ΔP_perd = {mejor_diseno.delta_p_perdida_total_pa:.2f} Pa)."
        )
    else:
        texto_columna = (
            f"{mejor_diseno.n_tubos} columnas tubulares de {mejor_diseno.diametro_nominal} con altura de "
            f"{mejor_diseno.altura_adoptada_m:.2f} metros cada una (Longitud total de caño: {long_total_m:.2f} m, "
            f"ΔP_tiro = {mejor_diseno.delta_p_tiro_pa:.2f} Pa > ΔP_perd = {mejor_diseno.delta_p_perdida_total_pa:.2f} Pa)."
        )

    return ConstructiveRecommendationResult(
        area_requerida_tambores_m2=area_req_tambores_m2,
        u_tambor_w_m2k=u_tambor,
        area_unitaria_tambor_m2=area_efectiva_tambor_m2,
        n_tambores_exacto=n_tambores_exacto,
        n_tambores_recomendados=n_tambores_rec,
        area_instalada_tambores_m2=area_inst_tambores,
        holgura_area_tambores_pct=holgura_tambores_pct,
        disposicion_tambores=disp_tambores,
        recomendacion_texto_tambores=texto_tambores,
        diametro_optimo_recomendado=mejor_diseno.diametro_nominal,
        n_columnas_recomendadas=mejor_diseno.n_tubos,
        altura_por_columna_recomendada_m=mejor_diseno.altura_adoptada_m,
        longitud_total_cano_m=long_total_m,
        area_instalada_columnas_m2=mejor_diseno.area_adoptada_total_m2,
        delta_p_tiro_optimo_pa=mejor_diseno.delta_p_tiro_pa,
        delta_p_perdida_optimo_pa=mejor_diseno.delta_p_perdida_total_pa,
        velocidad_optima_m_s=mejor_diseno.velocidad_interior_m_s,
        t_salida_optima_c=mejor_diseno.t_salida_estimada_con_altura_adoptada_c,
        recomendacion_texto_columna=texto_columna,
        diseno_optimo_detalle=mejor_diseno,
    )


def comparar_diametros_comerciales(
    m_dot_gas_kg_s: float,
    m_dot_cond_kg_s: float,
    material: str = "Cobre (Cu)",
    n_tubos: int = 1,
    t_entrada_c: float = 160.0,
    t_saturacion_c: float = 100.0,
    t_salida_diseno_c: float = 40.0,
    t_ambiente_c: float = 25.0,
    modo_calculo_u: str = "Churchill-Chu",
    u_fijo_w_m2k: float = 10.0,
    incluir_radiacion_externa: bool = False,
) -> pd.DataFrame:
    """
    Evalúa todos los diámetros comerciales del catálogo para las mismas condiciones
    de operación y devuelve una tabla comparativa de ingeniería.
    """
    filas: List[Dict[str, Any]] = []
    for diam_nom in TUBERIAS_COMERCIALES.keys():
        res = disenar_condensador_pasivo(
            m_dot_gas_kg_s=m_dot_gas_kg_s,
            m_dot_cond_kg_s=m_dot_cond_kg_s,
            diametro_nominal=diam_nom,
            material=material,
            n_tubos=n_tubos,
            t_entrada_c=t_entrada_c,
            t_saturacion_c=t_saturacion_c,
            t_salida_diseno_c=t_salida_diseno_c,
            t_ambiente_c=t_ambiente_c,
            modo_calculo_u=modo_calculo_u,
            u_fijo_w_m2k=u_fijo_w_m2k,
            incluir_radiacion_externa=incluir_radiacion_externa,
        )
        filas.append(
            {
                "Diámetro Nominal": diam_nom,
                "D_int (mm)": round(res.d_int_m * 1000.0, 1),
                "U Global (W/m²·K)": round(res.u_global_w_m2k, 2),
                "Área Total Req. (m²)": round(res.area_total_requerida_m2, 3),
                "Altura H por Tubo (m)": round(res.longitud_requerida_por_tubo_m, 2),
                "Velocidad Humos (m/s)": round(res.velocidad_interior_m_s, 3),
                "Reynolds (Re)": round(res.reynolds_interno, 0),
                "ΔP Tiro Natural (Pa)": round(res.delta_p_tiro_pa, 3),
                "ΔP Pérdidas (Pa)": round(res.delta_p_perdida_total_pa, 3),
                "Margen Neto (Pa)": round(res.margen_tiro_neto_pa, 3),
                "Estado": "✅ Apto" if (res.cumple_tiro_natural and res.cumple_temp_salida_45c) else "⚠️ Riesgo / No Apto",
            }
        )
    return pd.DataFrame(filas)


def generar_perfil_axial_temperatura(
    diseno: CondenserDesignResult, n_puntos: int = 50
) -> pd.DataFrame:
    """Genera el perfil axial de temperatura de los gases/vapores a lo largo de la columna."""
    z = np.linspace(0.0, diseno.altura_adoptada_m, n_puntos)
    frac_z = z / max(diseno.altura_adoptada_m, 1e-6)

    tau_axial = -math.log(
        max(
            (diseno.t_salida_estimada_con_altura_adoptada_c - diseno.t_ambiente_c)
            / max(diseno.t_entrada_c - diseno.t_ambiente_c, 1.0),
            0.02,
        )
    )
    t_fluido = diseno.t_ambiente_c + (diseno.t_entrada_c - diseno.t_ambiente_c) * np.exp(
        -tau_axial * frac_z
    )
    t_pared = diseno.t_ambiente_c + 0.78 * (t_fluido - diseno.t_ambiente_c)

    return pd.DataFrame(
        {
            "Altura Axial z (m)": z,
            "Temperatura Fluido Interno (°C)": t_fluido,
            "Temperatura Pared Tubo (°C)": t_pared,
            "Umbral Máximo Salida (45 °C)": np.full_like(z, 45.0),
            "Temperatura Ambiente (°C)": np.full_like(z, diseno.t_ambiente_c),
        }
    )
