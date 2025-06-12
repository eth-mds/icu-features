from pathlib import Path

VARIABLE_REFERENCE_PATH = Path(__file__).parents[1] / "resources" / "variables.tsv"

HORIZONS = [8, 24]

CAT_MISSING_NAME = "(MISSING)"

# Top variables according to fig 8a of Lyu et al 2024: An empirical study on
# KDIGO-defined acute kidney injury prediction in the intensive care unit.
KIDNEY_VARIABLES = [
    "time_hours",  # Time in hours since ICU admission
    "ufilt",  # Ultrafiltration on cont. RRT
    "ufilt_ind",  # Ultrafiltration on cont. RRT
    "rel_urine_rate",  # Urine rate per weight (ml/kg/h)
    "weight",
    "crea",  # Creatinine
    "etco2",  # End-tidal CO2
    "crp",  # C-reactive protein
    "anti_coag_ind",  # Indicator for antocoagulants treatment
    "hep",  # Heparin
    "hep_ind",  # Heparin
    "loop_diur",  # Loop diuretics
    "loop_diur_ind",  # Loop diuretics
    "resp",  # Respiratory rate
    "fluid_ind",  # Fluids
    "airway",  # Ventilation type
    "vent_ind",  # Indicator for any ventilation
    "bili",  # Bilirubin
    "anti_delir_ind",  # Indicator for antidelirium treatment
    "mg",  # Magnesium
    "op_pain_ind",  # Opioid pain medication
    "abx_ind",  # Antibiotics indicator
    "k",  # Potassium
]

# "preliminary selected variables" according to
# https://www.medrxiv.org/content/10.1101/2024.01.23.24301516v1 supp table 3
RESP_VARIABLES = [
    "fio2",
    "norepi",  # Norepinephrine
    "norepi_ind",  # Norepinephrine
    "dobu",  # Dobutamine
    "dobu_ind",  # Dobutamine
    "loop_diur",  # Loop diuretics
    "loop_diur_ind",  # Loop diuretics
    "benzdia",  # Benzodiazepines
    "benzdia_ind",  # Benzodiazepines
    "prop",  # Propofol
    "prop_ind",  # Propofol
    "ins_ind",  # Insulin
    "hep",  # Heparin
    "hep_ind",  # Heparin
    "cf_treat_ind",  # circulatory failure treatments incl. dobu, norepi.
    "sed_ind",  # sedation medication indicator incl. benzdia, prop.
    "age",
    # no emergency admission
    "vent_ind",  # Indicator for any ventilation
    "airway",  # Ventilation type
    "pco2",  # Partial pressure of carbon dioxide PaCO2
    "po2",  # Partial pressure of oxygen PaO2
    "sao2",  # Oxygen saturation (lab value) SaO2
    "spo2",  # Oxygen saturation (finger) SpO2
    "ps",  # Pressure support
    # No MV exp / MV spont. These are available in HiRID only
    "resp",  # Respiratory rate
    "supp_o2_vent",  # Oxygen supplementation
    "tgcs",  # Total Glasgow Coma Scale (Response)
    "mgcs",  # Motor Glasgow Coma Scale
    "peep",  # Positive end-expiratory pressure
    "map",  # Mean arterial pressure. ABPm is window-mean of map
    "peak",  # Peak airway pressure
    "ph",  # Used to determine po2 from sao2 according to the serveringhaus equation
    "temp",  # Temperature, used to determine po2 from sao2 according to serveringhaus
    "pf_ratio",  # ratio of po2 to fio2
]

# Top 20 variables of Hyland et al.: Early prediction of circulatory failure in the
# intensive care unit using machine learning. Table 1.
CIRC_VARIABLES = [
    "lact",  # Lactate
    "map",  # mean arterial pressure
    "time_hours",  # Time in hours since ICU admission
    "age",
    "hr",  # Heart rate
    "dobu",  # Dobutamine
    "dobu_ind",  # Dobutamine
    "milrin",  # Milrinone
    "milrin_ind",  # Milrinone
    "levo",  # Levosimendan
    "levo_ind",  # Levosimendan
    "teophyllin",  # Theophylline
    "teophyllin_ind",  # Theophylline
    "cf_treat_ind",  # circ. failure treatments incl. dobu, norepi, milrin, theo, levo
    "cout",  # Cardiac output
    "rass",  # Richmond Agitation Sedation Scale
    "inr_pt",  # Prothrombin
    "glu",  # Serum glucose
    "crp",  # C-reactive protein
    "dbp",  # Diastolic blood pressure
    "sbp",  # Systolic blood pressure
    "peak",  # Peak airway pressure
    "spo2",  # Oxygen saturation (finger) SpO2
    "nonop_pain_ind",  # Non-opioid pain medication
    "supp_o2_vent",  # Oxygen supplementation
]

# Variables used to determine apache II
APACHE_II_VARIABLES = [
    "age",
    "crea",
    "fio2",
    "hct",
    "hr",
    "k",
    "na",
    "pco2",
    "po2",
    "resp",
    "temp",
    "tgcs",
    "wbc",
]

CONTINUOUS_FEATURES = [
    "mean",
    "std",
    "slope",
    "fraction_nonnull",
    "all_missing",
    "min",
    "max",
]
CATEGORICAL_FEATURES = ["mode", "num_nonmissing"]
TREATMENT_INDICATOR_FEATURES = ["num", "any"]
TREATMENT_CONTINUOUS_FEATURES = ["rate"]
