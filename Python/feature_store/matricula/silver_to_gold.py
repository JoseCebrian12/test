import pandas as pd
import numpy as np
from ..general.funcs import load_parquet, delta_periodos
from ..general.dataframe_funcs import impute_near_value
import os

def processing_file_feature_level(df: pd.DataFrame, features_dict: dict = None) -> pd.DataFrame:
    '''
    Calculo de los features que se encuentran en la capa gold.
    Estos calculos se encuentran a nivel de PERIODO-COD_ALUMNO-FECHA_CORTE.
    '''

    df_ = df.copy()

    # parametors
    DICT_MODALIDADES, DICT_CAMPUS, DICT_FACULTADES, DICT_CARRERAS, DICT_TIPO_INGRESO = parameters()

    df_ = df_.sort_values(['periodo', 'cod_alumno'], ignore_index=True)

    # rellenar fecha de nacimiento
    # se calcula una estructura para determinar las fechas de nacimiento de cada alumno
    # si no se llega a imputar, se completa con la moda
    imp_table = df_.groupby(['cod_alumno', 'fecha_nacimiento'], as_index=False).agg(cant=('cod_alumno', 'count'))
    imp_table = imp_table.sort_values(by=['cod_alumno', 'cant'], ascending=[True, False])
    imp_table = imp_table.drop_duplicates(subset=['cod_alumno'], keep='first')
    df_ = df_.drop(['fecha_nacimiento'], axis=1)
    df_ = df_.merge(imp_table[['cod_alumno', 'fecha_nacimiento']], how='left', on=['cod_alumno'])
    df_.loc[:, 'fecha_nacimiento'] = df_.groupby('periodo')['fecha_nacimiento'].transform(lambda x: x.fillna(x.mode().iloc[0]))
    
    # fecha ingreso
    imp_table = df_.groupby(['cod_alumno', 'fecha_ingreso'], as_index=False).agg(cant=('cod_alumno', 'count'))
    imp_table = imp_table.sort_values(by=['cod_alumno', 'cant'], ascending=[True, False])
    imp_table = imp_table.drop_duplicates(subset=['cod_alumno'], keep='first')
    df_ = df_.drop(['fecha_ingreso'], axis=1)
    df_ = df_.merge(imp_table[['cod_alumno', 'fecha_ingreso']], how='left', on=['cod_alumno'])
    df_.loc[:, 'fecha_ingreso'] = df_['fecha_ingreso'].dt.normalize()

    # modalidad de estudio
    # se completa en funcion del estudiante y su historia
    # se prioriza a data del ciclo anterior
    # se completan los vacios con la moda por periodo
    var_name = 'modalidad_formativa'
    df_.loc[:, var_name] = impute_near_value(df_, ['cod_alumno'], var_name, 'periodo')

    # modalidad de estudio
    var_name = 'modalidad_estudio'
    df_.loc[:, var_name] = impute_near_value(df_, ['cod_alumno'], var_name, 'periodo')

    # modalidad de estudio
    var_name = 'campus'
    df_.loc[:, var_name] = impute_near_value(df_, ['cod_alumno'], var_name, 'periodo')

    # tipo de ingreso
    var_name = 'tipo_ingreso'
    df_.loc[:, var_name] = impute_near_value(df_, ['cod_alumno'], var_name, 'periodo')

    # departamento
    var_name = 'departamento'
    df_.loc[:, var_name] = impute_near_value(df_, ['cod_alumno'], var_name, 'periodo')

    # provincia
    var_name = 'provincia'
    df_.loc[:, var_name] = impute_near_value(df_, ['cod_alumno'], var_name, 'periodo')

    # distrito
    var_name = 'distrito'
    df_.loc[:, var_name] = impute_near_value(df_, ['cod_alumno'], var_name, 'periodo')

    # unidad de negocio
    df_.loc[:, 'unidad_negocio'] = np.where(df_['modalidad_estudio'] == 'PREGRADO-UPC', 'PG', 
                                            np.where(df_['carrera'].str.contains('WS'), 'WS', 'EPE'))
    
    # Mapear códigos de modalidad, campus, facultad, carrera y tipo de ingreso a descripciones
    # estos campos no se imputan porque siempre llegan completos
    df_.loc[:, 'modalidad'] = df_['modalidad_formativa'].map(DICT_MODALIDADES).fillna('NO_DETERMINADO')
    df_.loc[:, 'campus_desc'] = df_['campus'].map(DICT_CAMPUS).fillna('NO_DETERMINADO')
    df_.loc[:, 'facultad_desc'] = df_['facultad'].map(DICT_FACULTADES).fillna('NO_DETERMINADO')
    df_.loc[:, 'carrera_desc'] = df_['carrera'].map(DICT_CARRERAS).fillna('NO_DETERMINADO')
    df_.loc[:, 'grupo_tipo_ingreso'] = df_['tipo_ingreso'].map(DICT_TIPO_INGRESO).fillna('NO_DETERMINADO')

    # Calcular año, verificar si el periodo es de verano y calcular la edad
    df_.loc[:, 'anio'] = df_['periodo'] // 100
    df_.loc[:, 'es_verano'] = df_['periodo'].astype(str).str.endswith('00').astype(int)
    df_.loc[:, 'edad'] = (df_['anio'] - df_['fecha_nacimiento'].dt.year)

    # Normalizar porcentaje de beca y crear indicador de beca
    df_.loc[:, 'pct_beca'] = df_['porcentaje_beca'] / 100
    df_.loc[:, 'pct_beca'] = df_['pct_beca'].fillna(0)
    df_.loc[:, 'flag_beca'] = (df_['pct_beca'] != 0).astype(int)

    # Conteo del número de matrículas
    df_.loc[:, 'nro_matricula'] = 1
    df_.loc[:, 'n_per_mtr_hist'] = df_.groupby(['cod_alumno'])['nro_matricula'].cumsum()

    # conteo de matriculas regulares
    df_.loc[:, 'nro_matricula_reg'] = np.where(df_['es_verano'] == 1, 0, 1)
    df_.loc[:, 'n_per_mtr_hist_reg'] = df_.groupby(['cod_alumno'])['nro_matricula_reg'].cumsum()

    # conteo de matriculas verano
    df_.loc[:, 'nro_matricula_verano'] = np.where(df_['es_verano'] == 1, 1, 0)
    df_.loc[:, 'n_per_mtr_hist_verano'] = df_.groupby(['cod_alumno'])['nro_matricula_verano'].cumsum()

    # se matriculo en verano
    df_.loc[:, 'flag_mtr_verano'] = df_.groupby(['anio', 'cod_alumno'])['es_verano'].transform('any').astype(int)

    # para el estado de matricula
    # solo en funcion de los alumnos regulares
    imp_table = df_[['periodo', 'cod_alumno', 'es_verano']].copy()
    imp_table = imp_table[imp_table['es_verano'] == 0]
    imp_table.loc[:, 'periodo_b1'] = imp_table.groupby('cod_alumno')['periodo'].shift(1)
    imp_table.loc[:, 'periodo_b2'] = imp_table.groupby('cod_alumno')['periodo'].shift(2)
    imp_table.loc[:, 'periodo_b3'] = imp_table.groupby('cod_alumno')['periodo'].shift(3)
    imp_table.loc[:, 'periodo_b4'] = imp_table.groupby('cod_alumno')['periodo'].shift(4)
    imp_table.loc[:, 'periodo_b5'] = imp_table.groupby('cod_alumno')['periodo'].shift(5)
    imp_table.loc[:, 'periodo_b6'] = imp_table.groupby('cod_alumno')['periodo'].shift(6)
    imp_table.loc[:, 'periodo_b7'] = imp_table.groupby('cod_alumno')['periodo'].shift(7)
    imp_table.loc[:, 'periodo_b8'] = imp_table.groupby('cod_alumno')['periodo'].shift(8)
    imp_table.loc[:, 'periodo_b9'] = imp_table.groupby('cod_alumno')['periodo'].shift(9)
    imp_table.loc[:, 'periodo_b10'] = imp_table.groupby('cod_alumno')['periodo'].shift(10)

    imp_table.loc[:, 'delta_b1'] = delta_periodos(imp_table, 'periodo', 'periodo_b1', 2) # no se consideran veranos
    imp_table.loc[:, 'delta_b2'] = delta_periodos(imp_table, 'periodo_b1', 'periodo_b2', 2) # no se consideran veranos
    imp_table.loc[:, 'delta_b3'] = delta_periodos(imp_table, 'periodo_b2', 'periodo_b3', 2) # no se consideran veranos
    imp_table.loc[:, 'delta_b4'] = delta_periodos(imp_table, 'periodo_b3', 'periodo_b4', 2) # no se consideran veranos
    imp_table.loc[:, 'delta_b5'] = delta_periodos(imp_table, 'periodo_b4', 'periodo_b5', 2) # no se consideran veranos
    imp_table.loc[:, 'delta_b6'] = delta_periodos(imp_table, 'periodo_b5', 'periodo_b6', 2) # no se consideran veranos
    imp_table.loc[:, 'delta_b7'] = delta_periodos(imp_table, 'periodo_b6', 'periodo_b7', 2) # no se consideran veranos
    imp_table.loc[:, 'delta_b8'] = delta_periodos(imp_table, 'periodo_b7', 'periodo_b8', 2) # no se consideran veranos
    imp_table.loc[:, 'delta_b9'] = delta_periodos(imp_table, 'periodo_b8', 'periodo_b9', 2) # no se consideran veranos
    imp_table.loc[:, 'delta_b10'] = delta_periodos(imp_table, 'periodo_b9', 'periodo_b10', 2) # no se consideran veranos

    imp_table['delta_b1'] = imp_table['delta_b1'].fillna(0).astype(int).astype(str)
    imp_table['delta_b2'] = imp_table['delta_b2'].fillna(0).astype(int).astype(str)
    imp_table['delta_b3'] = imp_table['delta_b3'].fillna(0).astype(int).astype(str)
    imp_table['delta_b4'] = imp_table['delta_b4'].fillna(0).astype(int).astype(str)
    imp_table['delta_b5'] = imp_table['delta_b5'].fillna(0).astype(int).astype(str)
    imp_table['delta_b6'] = imp_table['delta_b6'].fillna(0).astype(int).astype(str)
    imp_table['delta_b7'] = imp_table['delta_b7'].fillna(0).astype(int).astype(str)
    imp_table['delta_b8'] = imp_table['delta_b8'].fillna(0).astype(int).astype(str)
    imp_table['delta_b9'] = imp_table['delta_b9'].fillna(0).astype(int).astype(str)
    imp_table['delta_b10'] = imp_table['delta_b10'].fillna(0).astype(int).astype(str)

    imp_table.loc[:, 'estado'] = imp_table['delta_b1'] + imp_table['delta_b2'] + imp_table['delta_b3'] + imp_table['delta_b4'] + imp_table['delta_b5'] \
        + imp_table['delta_b6'] + imp_table['delta_b7'] + imp_table['delta_b8'] + imp_table['delta_b9'] + imp_table['delta_b10']
    
    imp_table.loc[:, 'estado_matricula_new'] = imp_table['estado'].apply(estados_matricula)

    # cruzamos el estado de la matricula con el df_ final
    df_ = df_.merge(imp_table[['cod_alumno', 'periodo', 'estado_matricula_new']], how='left', on=['cod_alumno', 'periodo'])
    df_['estado_matricula_new'] = df_['estado_matricula_new'].fillna('REG_VERANO')

    df_['periodo_b1'] = df_.groupby('cod_alumno')['periodo'].shift(1, fill_value=0).astype(int)
    df_['periodo_b2'] = df_.groupby('cod_alumno')['periodo'].shift(2, fill_value=0).astype(int)
    df_['periodo_b3'] = df_.groupby('cod_alumno')['periodo'].shift(3, fill_value=0).astype(int)
    df_['periodo_b4'] = df_.groupby('cod_alumno')['periodo'].shift(4, fill_value=0).astype(int)
    df_['periodo_b5'] = df_.groupby('cod_alumno')['periodo'].shift(5, fill_value=0).astype(int)
    df_['periodo_b6'] = df_.groupby('cod_alumno')['periodo'].shift(6, fill_value=0).astype(int)
    df_['periodo_b7'] = df_.groupby('cod_alumno')['periodo'].shift(7, fill_value=0).astype(int)
    df_['periodo_b8'] = df_.groupby('cod_alumno')['periodo'].shift(8, fill_value=0).astype(int)
    df_['periodo_b9'] = df_.groupby('cod_alumno')['periodo'].shift(9, fill_value=0).astype(int)
    df_['periodo_b10'] = df_.groupby('cod_alumno')['periodo'].shift(10, fill_value=0).astype(int)

    # Calcular diferencias de año 1-2-3
    df_["diff_anio_1"] = df_.apply(
        lambda row: (
            int(str(row["periodo"])[:4]) - int(str(row["periodo_b1"])[:4])
            if row["periodo_b1"] != 0
            else 0
        ),
        axis=1,
    )
    df_["diff_anio_2"] = df_.apply(
        lambda row: (
            int(str(row["periodo"])[:4]) - int(str(row["periodo_b2"])[:4])
            if row["periodo_b2"] != 0
            else 0
        ),
        axis=1,
    )
    df_["diff_anio_3"] = df_.apply(
        lambda row: (
            int(str(row["periodo"])[:4]) - int(str(row["periodo_b3"])[:4])
            if row["periodo_b3"] != 0
            else 0
        ),
        axis=1,
    )

    # Crear indicador de matrícula en periodos anteriores
    periodos_regulares = ("01", "02")
    df_["flag_mtr_per_ant"] = (
        ((df_["diff_anio_1"] == 1) & df_["periodo_b1"].astype(str).str.endswith(periodos_regulares)) |
        ((df_["diff_anio_2"] == 1) & df_["periodo_b2"].astype(str).str.endswith(periodos_regulares)) |
        ((df_["diff_anio_3"] == 1) & df_["periodo_b3"].astype(str).str.endswith(periodos_regulares))
    ).astype(int)

    # Calcular diferencia de ciclos
    df_["diff_ciclos"] = np.where(
        df_["periodo_b1"] != 0,
        (
            (
                df_["periodo"].astype(str).str[:4].astype(int)
                - df_["periodo_b1"].astype(str).str[:4].astype(int)
            )
            * 3
            + df_["periodo"].astype(str).str[-2:].astype(int)
            - df_["periodo_b1"].astype(str).str[-2:].astype(int)
        ),
        0,
    )

    # Crear indicador de continuación de matrícula
    df_["flg_contador"] = (
        (~df_["es_verano"])
        & (((df_["periodo_b1"] == 0) | (df_["diff_ciclos"] == 1)) | (df_["diff_ciclos"] >= 2))).astype(int)

    # Asignar grupos para contar matrículas continuas
    df_["group"] = (df_["diff_ciclos"] > 2).groupby(df_["cod_alumno"]).cumsum()

    # Contar el número máximo de matrículas continuas
    df_["max_mtr_continuas_reg"] = df_.groupby(["cod_alumno", "group"])["flg_contador"].cumsum()

    if features_dict is not None:
        df_ = df_[features_dict.keys()]

    return df_

def estados_matricula(cadena_estados: str) -> str:
    if cadena_estados.startswith('0000000000'):
        return 'NEW_00'
    elif cadena_estados.startswith('1111111111'):
        return 'REG_10'
    elif cadena_estados.startswith('111111111'):
        return 'REG_09'
    elif cadena_estados.startswith('11111111'):
        return 'REG_08'
    elif cadena_estados.startswith('1111111'):
        return 'REG_07'
    elif cadena_estados.startswith('111111'):
        return 'REG_06'
    elif cadena_estados.startswith('11111'):
        return 'REG_05'
    elif cadena_estados.startswith('1111'):
        return 'REG_04'
    elif cadena_estados.startswith('111'):
        return 'REG_03'
    elif cadena_estados.startswith('11'):
        return 'REG_02'
    elif cadena_estados.startswith('1'):
        return 'REG_01'
    elif cadena_estados.startswith('2'):
        return 'REI_01'
    elif cadena_estados.startswith('3'):
        return 'REI_02'
    elif cadena_estados.startswith('4'):
        return 'REI_03'
    elif cadena_estados.startswith('5'):
        return 'REI_04'
    elif cadena_estados.startswith('6'):
        return 'REI_05'
    else:
        return 'REI_XX'

def parameters() -> dict:
    # diccionarios de procesamiento
    # especificos para procesar matricula
    DICT_MODALIDADES = {
        '0 - DATO NO DECLARADO': 'NO_DECL',
        '1 - PRESENCIAL': 'PRES',
        '2 - SEMI-PRESENCIAL': 'SEMI',
        '3 - A DISTANCIA': 'DIST',
    }

    DICT_CAMPUS = {
        'MONTERRICO': 'M',
        'SAN MIGUEL': 'SM',
        'VILLA': 'V',
        'SAN ISIDRO': 'SI'
    }

    DICT_FACULTADES = {
        'INGENIERIA': 'ING',
        'NEGOCIOS': 'NEG',
        'PSICOLOGIA': 'PSI',
        'ARQUITECTURA': 'ARQ',
        'DERECHO': 'DER',
        'COMUNICACIONES': 'COM',
        'ARTES CONTEMPORANEAS': 'ART',
        'CIENCIAS HUMANAS': 'CHUM',
        'EDUCACION': 'EDU',
        'CIENCIAS DE LA SALUD': 'CSAL',
        'ADMINISTRACION EN HOTELERIA Y TURISMO': 'ADM',
        'ECONOMIA': 'ECO',
        'DISENO': 'DIS',
    }

    DICT_CARRERAS = {
        'ADMINISTRACION DE EMPRESAS': 'ADM_EMP',
        'INGENIERIA CIVIL-EPE': 'ING_CIVIL',
        'ARQUITECTURA': 'ARQ',
        'ADMINISTRACION Y GERENCIA DEL EMPRENDIMIENTO': 'ADM_GER_EMP',
        'ING. INDUSTRIAL': 'ING_IND',
        'INGENIERIA DE SISTEMAS DE INFORMACION': 'ING_SIST_INF',
        'INGENIERIA DE REDES Y COMUNICACIONES': 'ING_RED_COM',
        'ING.SISTEMAS': 'ING_SIST',
        'MARKETING': 'MARK',
        'ADMINISTRACION DE EMPRESAS FDS': 'ADM_EMP_FDS',
        'ADMINISTRACION Y RECURSOS HUMANOS': 'ADM_RRHH',
        'ADMINISTRACION Y NEGOCIOS INTERNACIONALES': 'ADM_NEG_INT',
        'CONTABILIDAD': 'CONT',
        'ADMINISTRACION Y NEGOCIOS DEL DEPORTE': 'ADM_NEG_DEP',
        'ADMINISTRACION Y FINANZAS': 'ADM_FIN',
        'INGENIERIA DE SOFTWARE': 'ING_SOFT',
        'COMUNICACION Y MARKETING': 'COM_MARK',
        'INGENIERIA INDUSTRIAL': 'ING_IND',
        'CONTABILIDAD Y ADMINISTRACION': 'CONT_ADM',
        'NEGOCIOS INTERNACIONALES': 'NEG_INT',
        'INGENIERIA CIVIL': 'ING_CIVIL',
        'COMUNICACION E IMAGEN EMPRESARIAL': 'COM_IMG_EMP',
        'INGENIERIA ELECTRONICA': 'ING_ELEC',
        'PSICOLOGIA': 'PSI',
        'INGENIERIA DE TELECOMUNICACIONES Y REDES': 'ING_TEL_RED',
        'INGENIERIA INDUSTRIAL FDS': 'ING_IND_FDS',
        'ADMINISTRACION Y MARKETING': 'ADM_MARK',
        'DISENO Y GESTION EN MODA': 'DIS_GES_MODA',
        'DISENO PROFESIONAL DE INTERIORES': 'DIS_PROF_INT',
        'ADMINISTRACION DE BANCA Y FINANZAS': 'ADM_BAN_FIN',
        'CIENCIAS DE LA COMPUTACION': 'CIEN_COMP',
        'DERECHO': 'DER',
        'COMUNICACION AUDIOVISUAL Y MEDIOS INTERACTIVOS': 'COM_AUD_MED_INT',
        'ECONOMIA GERENCIAL': 'ECO_GER',
        'MUSICA': 'MUS',
        'HOTELERIA Y ADMINISTRACION': 'HOT_ADM',
        'COMUNICACION Y PUBLICIDAD': 'COM_PUB',
        'ECONOMIA Y NEGOCIOS INTERNACIONALES': 'ECO_NEG_INT',
        'TURISMO Y ADMINISTRACION': 'TUR_ADM',
        'INGENIERIA DE GESTION EMPRESARIAL': 'ING_GES_EMP',
        'NUTRICION Y DIETETICA': 'NUT_DIET',
        'DISENO PROFESIONAL GRAFICO': 'DIS_PROF_GRAF',
        'INGENIERIA MECATRONICA': 'ING_MECAT',
        'ECONOMIA Y FINANZAS': 'ECO_FIN',
        'ADMINISTRACION Y AGRONEGOCIOS': 'ADM_AGRON',
        'COMUNICACION Y PERIODISMO': 'COM_PER',
        'MEDICINA': 'MED',
        'ECONOMIA Y DESARROLLO': 'ECO_DES',
        'INGENIERIA DE GESTION MINERA': 'ING_GES_MIN',
        'EDUCACION Y GESTION DEL APRENDIZAJE': 'EDU_GES_APREND',
        'TERAPIA FISICA': 'TER_FIS',
        'INGENIERIA AMBIENTAL': 'ING_AMB',
        'TRADUCCION E INTERPRETACION PROFESIONAL': 'TRA_INT',
        'ODONTOLOGIA': 'ODON',
        'GASTRONOMIA Y GESTION CULINARIA': 'GAST_GES_CUL',
        'COMUNICACION Y FOTOGRAFIA': 'COM_FOT',
        'ARTES ESCENICAS': 'ART_ESC',
        'MEDICINA VETERINARIA': 'MED_VET',
        'DERECHO (WS)': 'DER',
        'ADMINISTRACION Y NEGOCIOS INTERNACIONALES (WS)': 'ADM_NEG_INT',
        'ADMINISTRACION Y MARKETING (WS)': 'ADM_MARK',
        'INGENIERIA DE SISTEMAS DE INFORMACION (WS)': 'ING_SIST_INF',
        'INGENIERIA INDUSTRIAL (WS)': 'ING_IND',
        'CONTABILIDAD Y ADMINISTRACION (WS)': 'CONT_ADM',
        'INGENIERIA CIVIL (WS)': 'ING_CIVIL',
        'COMUNICACION AUDIOVISUAL Y MEDIOS INTERACT (UA)': 'COM_AUD_MED_INT',
        'COMUNICACION Y MARKETING (UA)': 'COM_MARK',
        'ARQUITECTURA - (UA)': 'ARQ',
        'ADMINISTRACION Y NEGOCIOS INTERNACIONALES (UA)': 'ADM_NEG_INT',
        'RELACIONES INTERNACIONALES': 'REL_INT',
        'CIENCIAS POLITICAS': 'CIEN_POL',
        'MENCIONES PROPIAS': 'MEN_PRO',
        'MENCIONES EGRESADOS ADMINISTRACION Y MARKETING': 'MEN_EGR_ADM_MARK',
        'NEGOCIOS INTERNACIONALES WS': 'NEG_INT',
        'MARKETING WS': 'MARK',
        'CONTABILIDAD WS': 'CONT',
        'ADMINISTRACION DE EMPRESAS (WS)': 'ADM_EMP',
        'CIENCIAS DE LA ACTIVIDAD FISICA Y EL DEPORTE': 'CIEN_ACT_FIS_DEP',
        'MENCIONES EGRESADOS ADM Y NEGOC INTERNACIONALES': 'MEN_EGR_ADM_NEG_INT',
        'DISENO INDUSTRIAL': 'DIS_IND',
        'INGENIERIA BIOMEDICA': 'ING_BIOMED',
        'ADMINISTRACION': 'ADM',
        'BIOLOGIA': 'BIO',
        'ADMINISTRACION Y RECURSOS HUMANOS (WS)': 'ADM_RRHH',
        'CONTABILIDAD Y FINANZAS': 'CONT_FIN',
        'INGENIERIA DE SISTEMAS (WS)': 'ING_SIST',
        'DERECHO EPE': 'DER',
        'ADMINISTRACION DE BANCA Y FINANZAS (WS)': 'ADM_BAN_FIN',
        'ADMINISTRACION DE HOTELERIA Y TURISMO': 'ADM_HOT_TUR',
    }

    DICT_TIPO_INGRESO = {
        'ACREDITACION POR COMPETENCIA LABORAL': 'ACRED_COMPET_LAB',
        'AD PE CONVENIO SIN DIPLOMA': 'AD_PE_CONV',
        'AD PE CONVENIO SIN DIPLOMA MEDICINA': 'AD_PE_CONV',
        'ADMISION 30+': 'ADMISION_30',
        'ADP REGULAR': 'ADP_REGULAR',
        'ADP REGULAR MED.': 'ADP_REGULAR',
        'ADP SELECCION PREFERENTE': 'ADP_SEL_PREF',
        'ADP SELECCION PREFERENTE MED.': 'ADP_SEL_PREF',
        'ALUMNO LIBRE': 'AL',
        'BECA BICENTENARIO': 'BECA_BCNT',
        'BECAS HIJO DE DOCENTES': 'BECAS_HD',
        'BECAS PRONABEC': 'BECAS_PRONABEC',
        'BECAS PRONABEC - D': 'BECAS_PRONABEC',
        'CICLO DE PREPARACION': 'CICLO_PREPARACION',
        'CON. INT. CON DIPLOMA MED.': 'CON_INT_DIP_MED',
        'CONV. INT CON DIPLOMA ARIZONA - D': 'CONV_INT_DIP_CD',
        'CONV. INT CON DIPLOMA ARIZONA - P': 'CONV_INT_DIP_CD',
        'CONV. INT. CON DIPLOMA': 'CONV_INT_DIP_CD',
        'CONV. INT. CON DIPLOMA ARIZONA - S': 'CONV_INT_DIP_CD',
        'CONV. INT. MEDICINA': 'CONV_INT_MEDICINA',
        'CONV. INT. SIN DIPLOMA ARIZONA - S': 'CONV_INT_SIN_DIP',
        'CONVENIO INT. CON DIPLOMA - D': 'CONV_INT_DIP_CD',
        'CONVENIO INT. CON DIPLOMA - P': 'CONV_INT_DIP_CD',
        'CONVENIO INT. CON DIPLOMA - S': 'CONV_INT_DIP_CD',
        'CONVENIO INT. SIN DIPLOMA - P': 'CONV_INT_SIN_DIP',
        'CONVENIO INT. SIN DIPLOMA - S': 'CONV_INT_SIN_DIP',
        'CONVENIO. INT. CON DIPLOMA MED - P': 'CONV_INT_DIP_CD',
        'CONVENIO. INT. SIN DIPLOMA MED - P': 'CONV_INT_SIN_DIP',
        'CONVENIOS': 'CONVEN',
        'EGRESADOS CIBERTEC': 'EGR_CIBERTEC',
        'EPE (SOLO ING. EST.PREVIOS NO AFIN)': 'EPE_ING_EST_PREV',
        'EPE EST. SUP. PREV. COMPLETOS AFIN NEG.': 'EPE_EST_SUP',
        'EPE EST. SUP. PREVIOS INC. INGENIERIAS': 'EPE_EST_SUP',
        'EPE EST. SUP. PREVIOS INCOMPLETOS FDS': 'EPE_EST_SUP',
        'EPE ESTUDIOS SUP. PREVIOS COMPLETOS': 'EPE_EST_SUP',
        'EPE ESTUDIOS SUP. PREVIOS COMPLETOS FDS': 'EPE_EST_SUP',
        'EPE ESTUDIOS SUP. PREVIOS INCOMPLETOS': 'EPE_EST_SUP',
        'EPE INST CON CONVENIO-CONVALIDACIONES': 'EPE_INST_CONV_CONV',
        'EPE SIN ESTUDIOS SUP. PREVIOS': 'EPE_SIN_EST_SUP_PREV',
        'EPE SIN ESTUDIOS SUP. PREVIOS FDS ADM': 'EPE_SIN_EST_SUP_PREV',
        'ESCUELA PRE': 'ESCUELA_PRE',
        'EVAL. INTEGRAL MEDICINA': 'EVAL_INT',
        'EVALUACION INTEGRAL - D': 'EVAL_INT',
        'EVALUACION INTEGRAL - P': 'EVAL_INT',
        'EVALUACION INTEGRAL - S': 'EVAL_INT',
        'EVALUACION INTEGRAL ARIZONA - D': 'EVAL_INT',
        'EVALUACION INTEGRAL ARIZONA - P': 'EVAL_INT',
        'EVALUACION INTEGRAL ARIZONA - S': 'EVAL_INT',
        'EVALUACION INTEGRAL NCUK': 'EVAL_INT',
        'EXON.GRADO/ TITULO MEDICINA - P': 'EXO_GRADO_TIT',
        'EXONER.GRADO TITULO - D': 'EXO_GRADO_TIT',
        'EXONER.GRADO TITULO - P': 'EXO_GRADO_TIT',
        'EXONER.GRADO TITULO - S': 'EXO_GRADO_TIT',
        'EXONER.GRADO TITULO ARIZONA - D': 'EXO_GRADO_TIT',
        'EXONER.GRADO TITULO ARIZONA - S': 'EXO_GRADO_TIT',
        'FORMACION COMPLEMENTARIA': 'FORM_COMPL',
        'GENERAL': 'GENERAL',
        'GENERAL MEDICINA - P': 'GENERAL',
        'LOS 10 MEJORES': 'TOP_10',
        'OUI': 'OUI',
        'PPU': 'PPU',
        'PREMED/ACAMED - P': 'PREMED_ACAMED_P',
        'PROPEDEUTICO MUSICA': 'PROP_MSC',
        'RENDIMIENTO PROGRESIVO': 'REND_PROGR',
        'RENDIMIENTO PROGRESIVO (PQ)': 'REND_PROGR',
        'SEL. PREFERENTE MEDICINA': 'SEL_PREF',
        'SELECCION PREFERENTE': 'SEL_PREF',
        'SELECCION PREFERENTE - D': 'SEL_PREF',
        'SELECCION PREFERENTE - P': 'SEL_PREF',
        'SELECCION PREFERENTE - S': 'SEL_PREF',
        'SELECCION PREFERENTE ARIZONA - D': 'SEL_PREF',
        'SELECCION PREFERENTE ARIZONA - P': 'SEL_PREF',
        'SELECCION PREFERENTE ARIZONA - S': 'SEL_PREF',
        'SELECCION PREFERENTE MEDICINA - P': 'SEL_PREF',
        'SELECCION PREFERENTE WS': 'SEL_PREF',
        'TRASLADO EXTERNO - D': 'TRSLD_EXT',
        'TRASLADO EXTERNO ARIZONA - D': 'TRSLD_EXT',
        'TRASLADO EXTERNO ARIZONA - P': 'TRSLD_EXT',
        'TRASLADO EXTERNO ARIZONA - S': 'TRSLD_EXT',
        'TRASLADO EXTERNO MEDICINA - P': 'TRSLD_EXT',
        'TRASLADO INSTITUTO - D': 'TRSLD_INST',
        'TRASLADO INSTITUTO - S': 'TRSLD_INST',
        'TRASLADOS - P': 'TRSLD',
        'TRASLADOS - S': 'TRSLD',
    }

    return DICT_MODALIDADES, DICT_CAMPUS, DICT_FACULTADES, DICT_CARRERAS, DICT_TIPO_INGRESO
