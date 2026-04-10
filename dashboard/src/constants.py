# Catálogos de INEGI para mejorar la legibilidad

ESTADOS = {
    1: "Aguascalientes", 2: "Baja California", 3: "Baja California Sur",
    4: "Campeche", 5: "Coahuila", 6: "Colima", 7: "Chiapas",
    8: "Chihuahua", 9: "Ciudad de México", 10: "Durango",
    11: "Guanajuato", 12: "Guerrero", 13: "Hidalgo", 14: "Jalisco",
    15: "México", 16: "Michoacán", 17: "Morelos", 18: "Nayarit",
    19: "Nuevo León", 20: "Oaxaca", 21: "Puebla", 22: "Querétaro",
    23: "Quintana Roo", 24: "San Luis Potosí", 25: "Sinaloa",
    26: "Sonora", 27: "Tabasco", 28: "Tamaulipas", 29: "Tlaxcala",
    30: "Veracruz", 31: "Yucatán", 32: "Zacatecas"
}

NIVELES_EDUCATIVOS = {
    1: "Sin instrucción / Preescolar",
    2: "Primaria",
    3: "Secundaria",
    4: "Preparatoria / Licenciatura o superior"
}

ESTADOS_CIVILES = {
    1: "Unión Libre",
    2: "Separado(a)",
    3: "Divorciado(a)",
    4: "Viudo(a)",
    5: "Soltero(a)"
}

# Paleta de colores para el dashboard
COLOR_MUJER = "#c026d3" # Fuchsia
COLOR_HOMBRE = "#38bdf8" # Sky blue
COLOR_ACCENT = "#818cf8" # Indigo

# Mapeo de nombres técnicos a nombres legibles para el usuario
LABELS_MAPEADAS = {
    'eda': 'Edad',
    'anios_esc': 'Años de Estudio',
    'hrsocup': 'Horas Trabajadas (Semana)',
    'ingocup': 'Ingreso Mensual',
    'es_mujer': 'Género',
    'ur': 'Zona de Residencia',
    'n_hij': 'Número de Hijos',
    'e_con': 'Estado Civil',
    'niv_ins': 'Nivel Académico',
    't_loc_tri': 'Tamaño de Localidad',
    'cve_ent': 'Estado (Entidad)',
    'es_formal': 'Estatus Laboral',
    'participa_laboral': 'Participación en la Economía',
    'transfer': 'Remesas y Apoyos',
    'ing_cor': 'Ingreso Total Mensual',
    'gasto_mon': 'Gasto Corriente',
    'ingtrab': 'Ingreso por Trabajo',
    'becas': 'Becas y Transferencias'
}
