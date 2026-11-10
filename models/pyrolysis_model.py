"""
Módulo de Balances de Masa, Acondicionamiento, Cinética Global y Diseño Inverso
por Objetivo de Producción para Madera de Quebracho Colorado (Schinopsis balansae).
TFI Ingeniería - Máquinas Térmicas.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


# =============================================================================
# CONSTANTES Y PROPIEDADES DE REFERENCIA DE CÁTEDRA (QUEBRACHO COLORADO)
# =============================================================================

QUEBRACHO_PROPS = {
    "nombre_comun": "Quebracho Colorado",
    "nombre_cientifico": "Schinopsis balansae",
    "formula_empirica": "CH₁.₆₇O₀.₆₃",
    "composicion_elemental_seca_pct": {
        "C": 50.2,
        "H": 7.0,
        "O": 42.0,
        "Cenizas": 0.8,
    },
    "pci_seco_kcal_kg": 4593.78,
    "pci_15pct_hum_kcal_kg": 4503.78,
    "densidad_real_seca_kg_m3": 1200.0,
    "coeficiente_apilado_Ca": 0.75,
    "densidad_aparente_estiba_kg_m3": 900.0,
    "densidad_bioinsumo_kg_l": 1.04,  # Densidad media ácido piroleñoso + alquitrán (kg/L)
    "rendimientos_teoricos_10h_pct": {
        "carbon_solido": 39.87,
        "acido_pirolenoso": 41.41,
        "alquitran_vegetal": 2.19,
        "liquido_total": 43.60,
        "gases_incondensables": 16.53,
    },
    "tiempo_referencia_h": 10.0,
}

# Caso experimental de referencia de laboratorio (Ensayo 2024)
ENSAYO_LAB_2024 = {
    "descripcion": "Ensayo de Laboratorio 2024 (Retorta con caño de 3/4\" - Sobrepresión reportada)",
    "masa_humeda_g": 88.5,
    "humedad_pct": 0.0,
    "masa_carbon_g": 62.5,
    "masa_liquido_g": 16.7,
    "tiempo_min": 60.0,
    "diametro_nominal_pulg": "3/4\"",
}

# Presets de Escalas de Reactor
ESCALAS_REACTOR = {
    "Retorta de Laboratorio (0,2 a 1 kg)": {
        "tipo": "masa",
        "min_val": 0.05,
        "max_val": 2.0,
        "default_val": 0.5,
        "step": 0.05,
        "unidad": "kg",
        "tiempo_default_h": 1.0,
        "humedad_default_pct": 12.0,
        "diametro_sugerido": "1\"",
        "n_tubos_sugerido": 1,
        "descripcion": "Escala de banco para ensayos gravimétricos y caracterización de condensados.",
    },
    "Tambor 200 L / Calderín Termotanque (40 a 150 kg)": {
        "tipo": "masa",
        "min_val": 20.0,
        "max_val": 200.0,
        "default_val": 90.0,
        "step": 5.0,
        "unidad": "kg",
        "tiempo_default_h": 6.0,
        "humedad_default_pct": 15.0,
        "diametro_sugerido": "2\"",
        "n_tubos_sugerido": 1,
        "descripcion": "Escala piloto / semi-industrial en retorta metálica vertical u horizontal.",
    },
    "Horno Media Naranja Artesanal (10 a 15 m³)": {
        "tipo": "volumen",
        "min_val": 5.0,
        "max_val": 20.0,
        "default_val": 12.0,
        "step": 0.5,
        "unidad": "m³",
        "tiempo_default_h": 48.0,
        "humedad_default_pct": 15.0,
        "diametro_sugerido": "3\"",
        "n_tubos_sugerido": 4,
        "descripcion": "Horno de mampostería tradicional para producción carbonera con recuperación en troneras.",
    },
}


@dataclass
class PyrolysisBalanceResult:
    """Resultados completos del acondicionamiento y balance de masa de pirólisis."""
    masa_humeda_kg: float
    humedad_pct: float
    masa_seca_kg: float
    masa_agua_humedad_kg: float

    masa_carbon_kg: float
    masa_liquido_pirolisis_kg: float
    masa_acido_pirolenoso_kg: float
    masa_alquitran_kg: float
    masa_liquido_total_con_humedad_kg: float
    masa_gas_incondensable_kg: float

    rendimiento_solido_pct: float
    rendimiento_liquido_pct: float
    rendimiento_acido_pirolenoso_pct: float
    rendimiento_alquitran_pct: float
    rendimiento_gaseoso_pct: float

    tiempo_pirolisis_min: float
    tiempo_pirolisis_h: float
    m_dot_gas_g_min: float
    m_dot_gas_kg_h: float
    m_dot_gas_kg_s: float
    m_dot_cond_seco_g_min: float
    m_dot_cond_seco_kg_h: float
    m_dot_cond_seco_kg_s: float
    m_dot_cond_total_g_min: float
    m_dot_cond_total_kg_h: float
    m_dot_cond_total_kg_s: float
    m_dot_efluente_total_g_min: float
    m_dot_efluente_total_kg_h: float

    pci_seco_kcal_kg: float
    pci_humedo_kcal_kg: float
    energia_biomasa_kcal: float
    densidad_aparente_bio_kg_m3: float
    densidad_aparente_carb_kg_m3: float
    volumen_estiba_m3: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InversePyrolysisResult:
    """
    Resultado del cálculo de Diseño Inverso:
    Madera cargada + Objetivo de Producción (Litros o % Captura) -> Caudales de Diseño.
    """
    # Datos de carga de biomasa
    modo_carga_madera: str  # "kg" o "m3"
    masa_humeda_kg: float
    volumen_estiba_m3: float
    humedad_pct: float
    masa_seca_kg: float
    masa_agua_humedad_kg: float
    tiempo_ciclo_h: float
    tiempo_ciclo_min: float

    # Potencial estequiométrico teórico (100% de conversión de la hornada)
    masa_carbon_teor_kg: float
    masa_pirolenoso_teor_kg: float
    masa_alquitran_teor_kg: float
    masa_cond_pirolisis_teor_kg: float
    masa_cond_total_teor_kg: float  # Incluye humedad de la madera si corresponde
    volumen_cond_total_teor_litros: float
    masa_gas_teor_kg: float
    densidad_bioinsumo_kg_l: float

    # Meta solicitada vs. Meta factible (Comprobación de Saturación Estequiométrica)
    modo_objetivo: str  # "litros" o "captura_pct"
    litros_solicitados_usuario: float
    eficiencia_captura_solicitada_pct: float
    saturacion_estequiometrica: bool
    mensaje_viabilidad: str
    madera_humeda_minima_para_meta_kg: float
    volumen_madera_minimo_para_meta_m3: float

    # Valores efectivos de diseño adoptados para el condensador
    litros_objetivo_efectivo: float
    masa_cond_objetivo_kg: float
    masa_pirolenoso_objetivo_kg: float
    masa_alquitran_objetivo_kg: float
    masa_agua_humedad_objetivo_kg: float
    eficiencia_captura_efectiva_pct: float

    # Caudales másicos de diseño hacia el sistema de condensación
    m_dot_cond_objetivo_kg_h: float
    m_dot_cond_objetivo_g_min: float
    m_dot_cond_objetivo_kg_s: float
    litros_por_hora_l_h: float
    m_dot_gas_asociado_kg_h: float
    m_dot_gas_asociado_g_min: float
    m_dot_gas_asociado_kg_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calcular_humedad_base_humeda(masa_humeda: float, masa_seca: float) -> float:
    """Calcula el porcentaje de humedad en base húmeda (%H)."""
    if masa_humeda <= 0:
        raise ValueError("La masa húmeda debe ser mayor a cero.")
    if masa_seca > masa_humeda:
        raise ValueError("La masa seca no puede ser mayor que la masa húmeda inicial.")
    return ((masa_humeda - masa_seca) / masa_humeda) * 100.0


def calcular_masa_seca(masa_humeda: float, humedad_pct: float) -> float:
    """Calcula la masa de biomasa seca a partir de la masa húmeda y %H."""
    if not (0.0 <= humedad_pct < 100.0):
        raise ValueError("El porcentaje de humedad debe estar entre 0% y 99,9%.")
    return masa_humeda * (1.0 - (humedad_pct / 100.0))


def calcular_densidad_aparente(masa_kg: float, volumen_recipiente_m3: float) -> float:
    """Calcula la densidad aparente de la biomasa o del carbón."""
    if volumen_recipiente_m3 <= 0:
        raise ValueError("El volumen del recipiente debe ser mayor a cero.")
    return masa_kg / volumen_recipiente_m3


def calcular_pci_dulong(humedad_pct: float) -> float:
    """
    Calcula el Poder Calorífico Inferior (PCI) en kcal/kg ajustado por humedad (Dulong):
    PCI_seco = 4593,78 kcal/kg ; Con 15% de humedad: PCI_hum = 4503,78 kcal/kg.
    """
    pci_seco = QUEBRACHO_PROPS["pci_seco_kcal_kg"]
    return pci_seco - 600.0 * (humedad_pct / 100.0)


def estimar_rendimientos_por_temperatura(temp_horno_c: float) -> Dict[str, float]:
    """
    Modela la variación de los rendimientos de pirólisis lenta del Quebracho Colorado
    en el rango de 350 °C a 550 °C, anclado en los datos de cátedra a 400 °C
    (Carbón = 39,87%, Líquido = 43,60%, Gas = 16,53%).
    """
    t = float(np.clip(temp_horno_c, 300.0, 650.0))
    dt = t - 400.0

    r_solido = 39.87 - 0.072 * dt + 0.00008 * (dt ** 2)
    r_liquido = 43.60 + 0.014 * dt - 0.00021 * (dt ** 2)
    r_gaseoso = 100.0 - (r_solido + r_liquido)

    frac_pirolenoso = 41.41 / 43.60
    frac_alquitran = 2.19 / 43.60

    return {
        "r_solido_pct": round(r_solido, 2),
        "r_liquido_pct": round(r_liquido, 2),
        "r_acido_pirolenoso_pct": round(r_liquido * frac_pirolenoso, 2),
        "r_alquitran_pct": round(r_liquido * frac_alquitran, 2),
        "r_gaseoso_pct": round(r_gaseoso, 2),
    }


def resolver_diseno_inverso_produccion(
    valor_carga_madera: float,
    modo_carga_madera: str = "kg",  # "kg" o "m3"
    humedad_pct: float = 15.0,
    tiempo_ciclo_h: float = 48.0,
    modo_objetivo: str = "litros",  # "litros" o "captura_pct"
    litros_deseados: float = 400.0,
    eficiencia_captura_pct: float = 15.0,
    r_solido_pct: float = 39.87,
    r_liquido_pct: float = 43.60,
    incluir_humedad_en_bioinsumo: bool = True,
    densidad_bioinsumo_kg_l: float = 1.04,
    coef_apilado: float = 0.75,
) -> InversePyrolysisResult:
    """
    Resuelve el flujo de Diseño Inverso:
    1. A partir de la carga de madera (kg o m³) y %H, calcula la masa de volátiles
       condensables teóricos al 100% (M_cond_teor).
    2. Verifica la viabilidad frente al objetivo del usuario (Opción A: Litros deseados,
       Opción B: % de captura η). Si supera el 100% físico, emite alerta de saturación
       estequiométrica e informa cuánta madera mínima se requiere.
    3. Determina el caudal másico de condensación objetivo (m_dot_cond = M_cond_objetivo / Δt)
       y el caudal de gases incondensables asociado para dimensionar el condensador.
    """
    if tiempo_ciclo_h <= 0:
        raise ValueError("El tiempo estimado de la quema debe ser mayor a 0 horas.")
    if densidad_bioinsumo_kg_l <= 0:
        raise ValueError("La densidad del bioinsumo debe ser mayor a 0 kg/L.")

    rho_ap_estiba = QUEBRACHO_PROPS["densidad_real_seca_kg_m3"] * coef_apilado

    if modo_carga_madera == "m3":
        volumen_estiba_m3 = max(valor_carga_madera, 1e-4)
        masa_humeda_kg = volumen_estiba_m3 * rho_ap_estiba
    else:
        masa_humeda_kg = max(valor_carga_madera, 1e-4)
        volumen_estiba_m3 = masa_humeda_kg / max(rho_ap_estiba, 1.0)

    masa_seca_kg = calcular_masa_seca(masa_humeda_kg, humedad_pct)
    masa_agua_humedad_kg = masa_humeda_kg - masa_seca_kg

    # Rendimientos teóricos al 100% sobre base seca
    r_gas_pct = max(100.0 - (r_solido_pct + r_liquido_pct), 0.0)
    frac_pirolenoso = 41.41 / 43.60
    frac_alquitran = 2.19 / 43.60

    masa_carbon_teor_kg = masa_seca_kg * (r_solido_pct / 100.0)
    masa_cond_pirolisis_teor_kg = masa_seca_kg * (r_liquido_pct / 100.0)
    masa_pirolenoso_teor_kg = masa_cond_pirolisis_teor_kg * frac_pirolenoso
    masa_alquitran_teor_kg = masa_cond_pirolisis_teor_kg * frac_alquitran
    masa_gas_teor_kg = masa_seca_kg * (r_gas_pct / 100.0)

    if incluir_humedad_en_bioinsumo:
        masa_cond_total_teor_kg = masa_cond_pirolisis_teor_kg + masa_agua_humedad_kg
    else:
        masa_cond_total_teor_kg = masa_cond_pirolisis_teor_kg

    volumen_cond_total_teor_litros = masa_cond_total_teor_kg / densidad_bioinsumo_kg_l

    # Rendimiento líquido específico por cada kg de madera húmeda cargada (L / kg_húmedo)
    litros_por_kg_humedo = volumen_cond_total_teor_litros / max(masa_humeda_kg, 1e-9)

    # Evaluación del objetivo de producción (Opción A vs Opción B)
    if modo_objetivo == "litros":
        litros_solicitados = max(litros_deseados, 0.01)
        eta_solicitada_pct = (litros_solicitados / max(volumen_cond_total_teor_litros, 1e-9)) * 100.0
    else:
        eta_solicitada_pct = float(np.clip(eficiencia_captura_pct, 0.1, 100.0))
        litros_solicitados = volumen_cond_total_teor_litros * (eta_solicitada_pct / 100.0)

    madera_minima_kg = litros_solicitados / max(litros_por_kg_humedo, 1e-9)
    vol_madera_minimo_m3 = madera_minima_kg / max(rho_ap_estiba, 1.0)

    # Comprobación de viabilidad estequiométrica
    if eta_solicitada_pct > 100.001:
        saturacion = True
        litros_efectivos = volumen_cond_total_teor_litros
        eta_efectiva_pct = 100.0
        mensaje_viab = (
            f"⚠️ SATURACIÓN ESTEQUIOMÉTRICA: Solicitaste cosechar {litros_solicitados:,.1f} L, pero "
            f"la carga de {masa_humeda_kg:,.1f} kg de Quebracho ({volumen_estiba_m3:,.2f} m³) genera como máximo "
            f"físico {volumen_cond_total_teor_litros:,.1f} L al 100% de captura ({masa_cond_total_teor_kg:,.1f} kg). "
            f"Para obtener {litros_solicitados:,.1f} L al 100% necesitás cargar al menos "
            f"{madera_minima_kg:,.1f} kg ({vol_madera_minimo_m3:,.2f} m³) de leña."
        )
    else:
        saturacion = False
        litros_efectivos = litros_solicitados
        eta_efectiva_pct = eta_solicitada_pct
        mensaje_viab = (
            f"✅ META VIABLE ESTEQUIOMÉTRICAMENTE: Cosechar {litros_efectivos:,.1f} L equivale a "
            f"capturar el {eta_efectiva_pct:.1f}% del potencial condensable total de la hornada "
            f"({volumen_cond_total_teor_litros:,.1f} L máximos disponibles)."
        )

    factor_captura = eta_efectiva_pct / 100.0
    masa_cond_objetivo_kg = masa_cond_total_teor_kg * factor_captura
    masa_pirolenoso_obj_kg = masa_pirolenoso_teor_kg * factor_captura
    masa_alquitran_obj_kg = masa_alquitran_teor_kg * factor_captura
    masa_agua_obj_kg = (
        masa_agua_humedad_kg * factor_captura if incluir_humedad_en_bioinsumo else 0.0
    )

    # Caudales másicos hacia el sistema de condensación
    tiempo_min = tiempo_ciclo_h * 60.0
    tiempo_s = tiempo_ciclo_h * 3600.0

    m_dot_cond_kg_h = masa_cond_objetivo_kg / tiempo_ciclo_h
    m_dot_cond_g_min = (masa_cond_objetivo_kg * 1000.0) / tiempo_min
    m_dot_cond_kg_s = masa_cond_objetivo_kg / tiempo_s
    litros_por_hora = litros_efectivos / tiempo_ciclo_h

    # Caudal de gases incondensables arrastrados proporcionalmente por la corriente capturada
    masa_gas_asociada_kg = masa_gas_teor_kg * factor_captura
    m_dot_gas_kg_h = masa_gas_asociada_kg / tiempo_ciclo_h
    m_dot_gas_g_min = (masa_gas_asociada_kg * 1000.0) / tiempo_min
    m_dot_gas_kg_s = masa_gas_asociada_kg / tiempo_s

    return InversePyrolysisResult(
        modo_carga_madera=modo_carga_madera,
        masa_humeda_kg=masa_humeda_kg,
        volumen_estiba_m3=volumen_estiba_m3,
        humedad_pct=humedad_pct,
        masa_seca_kg=masa_seca_kg,
        masa_agua_humedad_kg=masa_agua_humedad_kg,
        tiempo_ciclo_h=tiempo_ciclo_h,
        tiempo_ciclo_min=tiempo_min,
        masa_carbon_teor_kg=masa_carbon_teor_kg,
        masa_pirolenoso_teor_kg=masa_pirolenoso_teor_kg,
        masa_alquitran_teor_kg=masa_alquitran_teor_kg,
        masa_cond_pirolisis_teor_kg=masa_cond_pirolisis_teor_kg,
        masa_cond_total_teor_kg=masa_cond_total_teor_kg,
        volumen_cond_total_teor_litros=volumen_cond_total_teor_litros,
        masa_gas_teor_kg=masa_gas_teor_kg,
        densidad_bioinsumo_kg_l=densidad_bioinsumo_kg_l,
        modo_objetivo=modo_objetivo,
        litros_solicitados_usuario=litros_solicitados,
        eficiencia_captura_solicitada_pct=eta_solicitada_pct,
        saturacion_estequiometrica=saturacion,
        mensaje_viabilidad=mensaje_viab,
        madera_humeda_minima_para_meta_kg=madera_minima_kg,
        volumen_madera_minimo_para_meta_m3=vol_madera_minimo_m3,
        litros_objetivo_efectivo=litros_efectivos,
        masa_cond_objetivo_kg=masa_cond_objetivo_kg,
        masa_pirolenoso_objetivo_kg=masa_pirolenoso_obj_kg,
        masa_alquitran_objetivo_kg=masa_alquitran_obj_kg,
        masa_agua_humedad_objetivo_kg=masa_agua_obj_kg,
        eficiencia_captura_efectiva_pct=eta_efectiva_pct,
        m_dot_cond_objetivo_kg_h=m_dot_cond_kg_h,
        m_dot_cond_objetivo_g_min=m_dot_cond_g_min,
        m_dot_cond_objetivo_kg_s=m_dot_cond_kg_s,
        litros_por_hora_l_h=litros_por_hora,
        m_dot_gas_asociado_kg_h=m_dot_gas_kg_h,
        m_dot_gas_asociado_g_min=m_dot_gas_g_min,
        m_dot_gas_asociado_kg_s=m_dot_gas_kg_s,
    )


def resolver_balance_experimental(
    masa_humeda_kg: float,
    humedad_pct: float,
    masa_carbon_kg: float,
    masa_liquido_kg: float,
    tiempo_pirolisis_min: float,
    volumen_recipiente_m3: Optional[float] = None,
    masa_bio_vol_kg: Optional[float] = None,
    masa_carb_vol_kg: Optional[float] = None,
    incluir_humedad_en_condensado: bool = False,
) -> PyrolysisBalanceResult:
    """
    Resuelve el balance gravimétrico en Modo Post-Ensayo (Laboratorio)
    a partir de las masas medidas experimentalmente.
    """
    if tiempo_pirolisis_min <= 0:
        raise ValueError("El tiempo de pirólisis debe ser mayor a cero.")

    masa_seca_kg = calcular_masa_seca(masa_humeda_kg, humedad_pct)
    masa_agua_humedad_kg = masa_humeda_kg - masa_seca_kg

    if masa_seca_kg <= 0:
        raise ValueError("La masa seca calculada debe ser mayor a cero.")

    r_solido_pct = (masa_carbon_kg / masa_seca_kg) * 100.0
    r_liquido_pct = (masa_liquido_kg / masa_seca_kg) * 100.0
    r_gaseoso_pct = 100.0 - (r_solido_pct + r_liquido_pct)

    masa_gas_kg = masa_seca_kg * (r_gaseoso_pct / 100.0)

    frac_pirolenoso = 41.41 / 43.60
    frac_alquitran = 2.19 / 43.60
    masa_pirolenoso_kg = masa_liquido_kg * frac_pirolenoso
    masa_alquitran_kg = masa_liquido_kg * frac_alquitran

    masa_liq_total_hum_kg = (
        masa_liquido_kg + masa_agua_humedad_kg
        if incluir_humedad_en_condensado
        else masa_liquido_kg
    )

    tiempo_h = tiempo_pirolisis_min / 60.0
    tiempo_s = tiempo_pirolisis_min * 60.0

    m_dot_gas_g_min = (masa_gas_kg * 1000.0) / tiempo_pirolisis_min
    m_dot_gas_kg_h = masa_gas_kg / tiempo_h
    m_dot_gas_kg_s = masa_gas_kg / tiempo_s

    m_dot_cond_seco_g_min = (masa_liquido_kg * 1000.0) / tiempo_pirolisis_min
    m_dot_cond_seco_kg_h = masa_liquido_kg / tiempo_h
    m_dot_cond_seco_kg_s = masa_liquido_kg / tiempo_s

    m_dot_cond_total_g_min = (masa_liq_total_hum_kg * 1000.0) / tiempo_pirolisis_min
    m_dot_cond_total_kg_h = masa_liq_total_hum_kg / tiempo_h
    m_dot_cond_total_kg_s = masa_liq_total_hum_kg / tiempo_s

    m_dot_efluente_total_g_min = m_dot_gas_g_min + m_dot_cond_total_g_min
    m_dot_efluente_total_kg_h = m_dot_gas_kg_h + m_dot_cond_total_kg_h

    if volumen_recipiente_m3 and volumen_recipiente_m3 > 0:
        rho_ap_bio = calcular_densidad_aparente(
            masa_bio_vol_kg if masa_bio_vol_kg is not None else masa_humeda_kg,
            volumen_recipiente_m3,
        )
        rho_ap_carb = calcular_densidad_aparente(
            masa_carb_vol_kg if masa_carb_vol_kg is not None else masa_carbon_kg,
            volumen_recipiente_m3,
        )
        vol_estiba = volumen_recipiente_m3
    else:
        rho_ap_bio = QUEBRACHO_PROPS["densidad_aparente_estiba_kg_m3"]
        rho_ap_carb = rho_ap_bio * (r_solido_pct / 100.0)
        vol_estiba = masa_seca_kg / rho_ap_bio

    pci_seco = QUEBRACHO_PROPS["pci_seco_kcal_kg"]
    pci_hum = calcular_pci_dulong(humedad_pct)

    return PyrolysisBalanceResult(
        masa_humeda_kg=masa_humeda_kg,
        humedad_pct=humedad_pct,
        masa_seca_kg=masa_seca_kg,
        masa_agua_humedad_kg=masa_agua_humedad_kg,
        masa_carbon_kg=masa_carbon_kg,
        masa_liquido_pirolisis_kg=masa_liquido_kg,
        masa_acido_pirolenoso_kg=masa_pirolenoso_kg,
        masa_alquitran_kg=masa_alquitran_kg,
        masa_liquido_total_con_humedad_kg=masa_liq_total_hum_kg,
        masa_gas_incondensable_kg=masa_gas_kg,
        rendimiento_solido_pct=r_solido_pct,
        rendimiento_liquido_pct=r_liquido_pct,
        rendimiento_acido_pirolenoso_pct=r_liquido_pct * frac_pirolenoso,
        rendimiento_alquitran_pct=r_liquido_pct * frac_alquitran,
        rendimiento_gaseoso_pct=r_gaseoso_pct,
        tiempo_pirolisis_min=tiempo_pirolisis_min,
        tiempo_pirolisis_h=tiempo_h,
        m_dot_gas_g_min=m_dot_gas_g_min,
        m_dot_gas_kg_h=m_dot_gas_kg_h,
        m_dot_gas_kg_s=m_dot_gas_kg_s,
        m_dot_cond_seco_g_min=m_dot_cond_seco_g_min,
        m_dot_cond_seco_kg_h=m_dot_cond_seco_kg_h,
        m_dot_cond_seco_kg_s=m_dot_cond_seco_kg_s,
        m_dot_cond_total_g_min=m_dot_cond_total_g_min,
        m_dot_cond_total_kg_h=m_dot_cond_total_kg_h,
        m_dot_cond_total_kg_s=m_dot_cond_total_kg_s,
        m_dot_efluente_total_g_min=m_dot_efluente_total_g_min,
        m_dot_efluente_total_kg_h=m_dot_efluente_total_kg_h,
        pci_seco_kcal_kg=pci_seco,
        pci_humedo_kcal_kg=pci_hum,
        energia_biomasa_kcal=masa_humeda_kg * pci_hum,
        densidad_aparente_bio_kg_m3=rho_ap_bio,
        densidad_aparente_carb_kg_m3=rho_ap_carb,
        volumen_estiba_m3=vol_estiba,
    )


def resolver_balance_predictivo(
    masa_humeda_kg: float,
    humedad_pct: float,
    tiempo_pirolisis_h: float,
    r_solido_pct: float = 39.87,
    r_liquido_pct: float = 43.60,
    incluir_humedad_en_condensado: bool = True,
    coef_apilado: float = 0.75,
) -> PyrolysisBalanceResult:
    """Resuelve el balance predictivo de masa al 100% sobre base seca."""
    masa_seca_kg = calcular_masa_seca(masa_humeda_kg, humedad_pct)
    masa_carbon_kg = masa_seca_kg * (r_solido_pct / 100.0)
    masa_liquido_kg = masa_seca_kg * (r_liquido_pct / 100.0)

    rho_ap_bio = QUEBRACHO_PROPS["densidad_real_seca_kg_m3"] * coef_apilado
    vol_estiba_m3 = masa_seca_kg / rho_ap_bio if rho_ap_bio > 0 else 0.0

    return resolver_balance_experimental(
        masa_humeda_kg=masa_humeda_kg,
        humedad_pct=humedad_pct,
        masa_carbon_kg=masa_carbon_kg,
        masa_liquido_kg=masa_liquido_kg,
        tiempo_pirolisis_min=tiempo_pirolisis_h * 60.0,
        volumen_recipiente_m3=vol_estiba_m3,
        masa_bio_vol_kg=masa_seca_kg,
        masa_carb_vol_kg=masa_carbon_kg,
        incluir_humedad_en_condensado=incluir_humedad_en_condensado,
    )


def generar_curva_temperatura_rendimiento(
    t_min_c: float = 350.0, t_max_c: float = 550.0, n_puntos: int = 41
) -> pd.DataFrame:
    """Genera un DataFrame con la evolución de los rendimientos entre 350 °C y 550 °C."""
    temperaturas = np.linspace(t_min_c, t_max_c, n_puntos)
    filas = []
    for temp in temperaturas:
        r = estimar_rendimientos_por_temperatura(float(temp))
        filas.append(
            {
                "Temperatura (°C)": float(temp),
                "Carbón Sólido (%)": r["r_solido_pct"],
                "Líquido Total (%)": r["r_liquido_pct"],
                "Ácido Piroleñoso (%)": r["r_acido_pirolenoso_pct"],
                "Alquitrán Vegetal (%)": r["r_alquitran_pct"],
                "Gases Incondensables (%)": r["r_gaseoso_pct"],
            }
        )
    return pd.DataFrame(filas)


def generar_curva_temporal_pirolisis(
    balance: PyrolysisBalanceResult, n_puntos: int = 50
) -> pd.DataFrame:
    """Genera el perfil temporal acumulado de conversión de biomasa."""
    t_min = np.linspace(0.0, balance.tiempo_pirolisis_min, n_puntos)
    tau = t_min / max(balance.tiempo_pirolisis_min, 1e-6)

    fraccion_avance = 1.0 / (1.0 + np.exp(-10.0 * (tau - 0.45)))
    f_min, f_max = fraccion_avance[0], fraccion_avance[-1]
    fraccion_avance = (fraccion_avance - f_min) / (f_max - f_min)

    return pd.DataFrame(
        {
            "Tiempo (min)": t_min,
            "Tiempo (h)": t_min / 60.0,
            "Avance de Pirólisis (%)": fraccion_avance * 100.0,
            "Biomasa Seca Remanente (kg)": balance.masa_seca_kg
            - fraccion_avance * (balance.masa_seca_kg - balance.masa_carbon_kg),
            "Líquido Condensado Acumulado (kg)": fraccion_avance
            * balance.masa_liquido_total_con_humedad_kg,
            "Gases Generados Acumulados (kg)": fraccion_avance
            * balance.masa_gas_incondensable_kg,
        }
    )
