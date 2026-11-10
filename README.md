# 🔥 Simulador de Pirólisis de Biomasa (Quebracho Colorado) y Condensación Pasiva por Tiro Natural

Aplicación web interactiva desarrollada en **Python + Streamlit** para el **Trabajo Final Integrador (TFI) de Ingeniería — Máquinas Térmicas**.

Permite simular, parametrizar y verificar el proceso de pirólisis lenta de madera de **Quebracho Colorado (*Schinopsis balansae*)** y el diseño termodinámico/fluidodinámico de una **columna condensadora pasiva por tiro natural (efecto chimenea)** con trampa separadora en **"Y"** para la recuperación de ácido piroleñoso y alquitrán vegetal.

---

## 📁 Estructura del Proyecto

```text
pirolisis/
├── app.py                      # Interfaz web interactiva en Streamlit (3 pestañas)
├── models/
│   ├── __init__.py
│   ├── pyrolysis_model.py      # Balances de masa, humedad, densidad aparente, PCI Dulong y rendimientos
│   └── thermal_model.py        # Balance térmico en 4 etapas, Churchill-Chu, tiro natural y pérdida de carga
├── tests/
│   └── test_models.py          # Pruebas de validación numérica de los modelos
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Documentación de uso
```

---

## 🚀 Instalación y Ejecución

1. **Instalar las dependencias** en tu entorno de Python (3.10+):
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar la aplicación interactiva en Streamlit**:
   ```bash
   streamlit run app.py
   ```

3. **Ejecutar las pruebas de verificación de los modelos**:
   ```bash
   python -m unittest discover -s tests
   ```

---

## 🛠️ Características Principales Implementadas

1. **Pestaña 1 — Modo Post-Ensayo (Planilla Experimental de Laboratorio):**
   - Incluye como caso precargado el **Ensayo de Laboratorio 2024** ($88,5\text{ g}$ de quebracho, $62,5\text{ g}$ de carbón, $16,7\text{ g}$ de líquido en $60\text{ min}$ con caño de $3/4''$), evidenciando el diagnóstico de riesgo de sobrepresión por estrangulamiento en diámetros menores a $1''$.
   - Permite cargar nuevos ensayos en gramos o kilogramos, calcular la humedad desde pesada de estufa y comparar rendimientos frente a los valores teóricos de cátedra.

2. **Pestaña 2 — Modo Predictivo y Escalamiento (Diseño del Reactor y Condensador):**
   - Tres escalas preconfiguradas:
     - **Retorta de Laboratorio** ($0,2\text{ a }1\text{ kg}$)
     - **Tambor de 200 L / Calderín de Termotanque** ($40\text{ a }150\text{ kg}$)
     - **Horno Media Naranja Artesanal** ($10\text{ a }15\text{ m}^3$ de leña estibada)
   - Desglose de potencia térmica a disipar ($\dot{Q}_{\text{total}}$) en sus **4 etapas**: enfriamiento de gases incondensables, desrecalentamiento de vapores, condensación latente ($h_{fg} = 2257\text{ kJ/kg}$) y subenfriamiento de líquido.
   - Selección de material (**Cobre, Aluminio, Acero al Carbono, Inoxidable**) y comparación automática de **diámetros comerciales ($3/4''$ a $4''$)** verificando que $\Delta P_{\text{tiro}} > \Delta P_{\text{perd}}$ y $T_s < 45\ ^\circ\text{C}$.

3. **Pestaña 3 — Marco Teórico, Ecuaciones y Esquema de Trampa en "Y":**
   - Compendio de ecuaciones termoquímicas y termohidráulicas y diagrama técnico del equipo.
