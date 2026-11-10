"""Pruebas unitarias para los modelos de pirólisis, diseño inverso y condensación pasiva."""

import unittest
from models.pyrolysis_model import (
    calcular_humedad_base_humeda,
    calcular_masa_seca,
    calcular_pci_dulong,
    resolver_diseno_inverso_produccion,
    resolver_balance_experimental,
    resolver_balance_predictivo,
)
from models.thermal_model import (
    disenar_condensador_pasivo,
    recomendar_configuracion_constructiva,
    comparar_diametros_comerciales,
)


class TestPyrolysisAndThermalModels(unittest.TestCase):
    def test_pci_dulong_catedra(self):
        self.assertAlmostEqual(calcular_pci_dulong(0.0), 4593.78, places=2)
        self.assertAlmostEqual(calcular_pci_dulong(15.0), 4503.78, places=2)

    def test_humedad_y_masa_seca(self):
        h = calcular_humedad_base_humeda(100.0, 85.0)
        self.assertAlmostEqual(h, 15.0, places=4)
        m_s = calcular_masa_seca(100.0, 15.0)
        self.assertAlmostEqual(m_s, 85.0, places=4)

    def test_diseno_inverso_400_litros_viable_y_recomendacion(self):
        # 12 m³ de Quebracho al 15% H en 48 h, pidiendo 400 L de bioinsumo
        inv = resolver_diseno_inverso_produccion(
            valor_carga_madera=12.0,
            modo_carga_madera="m3",
            humedad_pct=15.0,
            tiempo_ciclo_h=48.0,
            modo_objetivo="litros",
            litros_deseados=400.0,
        )
        self.assertFalse(inv.saturacion_estequiometrica)
        self.assertAlmostEqual(inv.litros_objetivo_efectivo, 400.0, places=2)
        self.assertGreater(inv.volumen_cond_total_teor_litros, 400.0)

        rec = recomendar_configuracion_constructiva(
            m_dot_gas_kg_s=inv.m_dot_gas_asociado_kg_s,
            m_dot_cond_kg_s=inv.m_dot_cond_objetivo_kg_s,
            litros_objetivo=inv.litros_objetivo_efectivo,
        )
        self.assertGreaterEqual(rec.n_tambores_recomendados, 1)
        self.assertTrue(rec.diseno_optimo_detalle.cumple_tiro_natural)
        self.assertTrue(rec.diseno_optimo_detalle.cumple_temp_salida_45c)

    def test_saturacion_estequiometrica_detectada(self):
        # Si cargamos solo 100 kg de madera y pedimos 400 L -> debe activar alerta de saturación
        inv_sat = resolver_diseno_inverso_produccion(
            valor_carga_madera=100.0,
            modo_carga_madera="kg",
            humedad_pct=15.0,
            tiempo_ciclo_h=6.0,
            modo_objetivo="litros",
            litros_deseados=400.0,
        )
        self.assertTrue(inv_sat.saturacion_estequiometrica)
        self.assertAlmostEqual(inv_sat.eficiencia_captura_efectiva_pct, 100.0, places=2)
        self.assertGreater(inv_sat.madera_humeda_minima_para_meta_kg, 100.0)

    def test_ensayo_lab_2024_y_sobrepresion_3_4(self):
        res_lab = resolver_balance_experimental(
            masa_humeda_kg=0.0885,
            humedad_pct=0.0,
            masa_carbon_kg=0.0625,
            masa_liquido_kg=0.0167,
            tiempo_pirolisis_min=60.0,
        )
        self.assertAlmostEqual(res_lab.masa_gas_incondensable_kg * 1000.0, 9.3, places=2)

        cond_3_4 = disenar_condensador_pasivo(
            m_dot_gas_kg_s=res_lab.m_dot_gas_kg_s,
            m_dot_cond_kg_s=res_lab.m_dot_cond_total_kg_s,
            diametro_nominal="3/4\"",
            altura_adoptada_m=1.0,
        )
        self.assertFalse(cond_3_4.cumple_tiro_natural)

        cond_1_in = disenar_condensador_pasivo(
            m_dot_gas_kg_s=res_lab.m_dot_gas_kg_s,
            m_dot_cond_kg_s=res_lab.m_dot_cond_total_kg_s,
            diametro_nominal="1\"",
            altura_adoptada_m=1.0,
        )
        self.assertTrue(cond_1_in.cumple_tiro_natural)
        self.assertTrue(cond_1_in.cumple_temp_salida_45c)


if __name__ == "__main__":
    unittest.main()
