import re

import numpy as np
import pandas as pd


# fmt: off
def processing_file_feature_level(df: pd.DataFrame, features_dict: dict = None) -> pd.DataFrame:
    """
    Calculo de los features que se encuentran en la capa gold.
    Estos calculos se encuentran a nivel de PERIODO-COD_ALUMNO-FECHA_CORTE.
    """

    # agrupaciones de evaluaciones
    EVAL_PRACTICAS = ["PC", "CL", "LB", "CC"] # PRÁCTICAS PC | CONTROL DE LECTURA | PRACTICA LABORATORIO | PROMEDIO DE CONTROLES
    EVAL_EXAMENES = ["EB", "EA", "NU"] # EVALUACIÓN FINAL | EVALUACIÓN PARCIAL | NOTA ÚNICA
    EVAL_TRABAJOS = ["TF", "TA", "TB", "TP"] # TRABAJO FINAL | TAREAS ACADÉMICAS | TRABAJO | TRABAJO PARCIAL
    EVAL_PARTICIPACION = ["DD", "PA", "EX", "EC", "CD", "ED"] # EVAL. DE DESEMPENO | PARTICIPACIÓN | EXPOSICIÓN | PROMEDIO EVALUACIÓN CONTINUA | PROMEDIO DE EVALUACIÓN DE DESE | EVALUACIÓN DE DESTREZAS

    df_ = df.copy().reset_index(drop=True)

    # eliminacion de notas no permitidas
    b1 = df_["nota"] <= 20.5
    b2 = df_["nota"] >= 0
    b3 = df_["nota"].isin([88, 99, 77, 55])   # {88: "NO_RINDIO", 77: "SIN_NOTA", 99: "SANCIONADO", 55: "RETIRADO"}
    df_ = df_[(b1 & b2) | b3]

    df_ = df_[~df_["nota"].isna()] # eliminamos registros donde la nota sea nula
    df_ = df_[df_["cod_tipo_prueba"] != "PF"].copy().reset_index(drop=True) # no consideramos promedis finales
    
    # flag de evaluaciones presentes
    df_.loc[:, "flag_eval_presente"] = np.where(df_["nota"] <= 20.5, 1, 0)

    # para las notas especiales se consideran 0.
    df_.loc[:, "nota_val"] = np.where(df_["nota"]>=20.5, 0, df_["nota"]) # el 20.5 es por un tema de decimales

    # aprobaciones
    df_.loc[:, "flag_aprobado"] = np.where(df_["nota_val"]>=13, 1, 0)
    df_.loc[:, "nota_aprob"] = df_["nota_val"] * df_["flag_aprobado"]

    # creamos los flags sobre tipos de notas, SOLO RENDIDAS
    df_.loc[:, "flag_practicas"] = np.where(df_["cod_tipo_prueba"].isin(EVAL_PRACTICAS), 1, 0)
    df_.loc[:, "flag_examenes"] = np.where(df_["cod_tipo_prueba"].isin(EVAL_EXAMENES), 1, 0)
    df_.loc[:, "flag_trabajos"] = np.where(df_["cod_tipo_prueba"].isin(EVAL_TRABAJOS), 1, 0)
    df_.loc[:, "flag_participacion"] = np.where(df_["cod_tipo_prueba"].isin(EVAL_PARTICIPACION), 1, 0)

    df_.loc[:, "flag_practicas"] = df_["flag_practicas"] * df_["flag_eval_presente"] 
    df_.loc[:, "flag_examenes"] = df_["flag_examenes"] * df_["flag_eval_presente"]
    df_.loc[:, "flag_trabajos"] = df_["flag_trabajos"] * df_["flag_eval_presente"]
    df_.loc[:, "flag_participacion"] = df_["flag_participacion"] * df_["flag_eval_presente"]

    # flags de tipos de pruebas aprobadas
    df_.loc[:, "flag_practicas_aprob"] = df_["flag_practicas"] * df_["flag_aprobado"]
    df_.loc[:, "flag_examenes_aprob"] = df_["flag_examenes"] * df_["flag_aprobado"]
    df_.loc[:, "flag_trabajos_aprob"] = df_["flag_trabajos"] * df_["flag_aprobado"]
    df_.loc[:, "flag_participacion_aprob"] = df_["flag_participacion"] * df_["flag_aprobado"]

    # creamos los valores de notas sobre tipos de notas
    df_.loc[:, "nota_practicas"] = df_["nota_val"] * df_["flag_practicas"]
    df_.loc[:, "nota_examenes"] = df_["nota_val"] * df_["flag_examenes"]
    df_.loc[:, "nota_trabajos"] = df_["nota_val"] * df_["flag_trabajos"]
    df_.loc[:, "nota_participacion"] = df_["nota_val"] * df_["flag_participacion"]

    # creamos los valores de notas sobre tipos de notas solo aprobados
    df_.loc[:, "nota_practicas_aprob"] = df_["nota_val"] * df_["flag_practicas_aprob"]
    df_.loc[:, "nota_examenes_aprob"] = df_["nota_val"] * df_["flag_examenes_aprob"]
    df_.loc[:, "nota_trabajos_aprob"] = df_["nota_val"] * df_["flag_trabajos_aprob"]
    df_.loc[:, "nota_participacion_aprob"] = df_["nota_val"] * df_["flag_participacion_aprob"]

    # numero de pruebas esperadas
    df_.loc[:, "evals_esperadas"] = df_["formula"].apply(lambda x:len(str(x).split("+")))
    

    # creamos los features de la capa gold
    df_ = df_.groupby(["periodo", "cod_alumno", "fecha_corte"], as_index=False, dropna=False).agg(
        n_eval_total_avg = ("evals_esperadas", lambda x: round(x.mean(), 0)),
        n_cursos = ("cod_curso", "nunique"),

        n_eval = ("nota_val", "count"),
        n_eval_asist = ("flag_eval_presente", "sum"),
        n_eval_aprob = ("flag_aprobado", "sum"),
        
        n_eval_practicas = ("flag_practicas", "sum"),
        n_eval_examenes = ("flag_examenes", "sum"),
        n_eval_trabajos = ("flag_trabajos", "sum"),
        n_eval_participacion = ("flag_participacion", "sum"),
        
        n_eval_practicas_aprob = ("flag_practicas_aprob", "sum"),
        n_eval_examenes_aprob = ("flag_examenes_aprob", "sum"),
        n_eval_trabajos_aprob = ("flag_trabajos_aprob", "sum"),
        n_eval_participacion_aprob = ("flag_participacion_aprob", "sum"),
        
        sum_notas_eval_total = ("nota_val", "sum"),
        sum_notas_eval_aprob = ("nota_aprob", "sum"),
        
        sum_notas_eval_practicas = ("nota_practicas", "sum"),
        sum_notas_eval_examenes = ("nota_examenes", "sum"),
        sum_notas_eval_trabajos = ("nota_trabajos", "sum"),
        sum_notas_eval_participacion = ("nota_participacion", "sum"),
        
        sum_notas_eval_practicas_aprob = ("nota_practicas_aprob", "sum"),
        sum_notas_eval_examenes_aprob = ("nota_examenes_aprob", "sum"),
        sum_notas_eval_trabajos_aprob = ("nota_trabajos_aprob", "sum"),
        sum_notas_eval_participacion_aprob = ("nota_participacion_aprob", "sum"),
    )

    # ratios
    df_.loc[:, "r_eval_asist"] = df_["n_eval_asist"] / df_["n_eval"] 
    df_.loc[:, "r_eval_aprob"] = df_["n_eval_aprob"] / df_["n_eval_asist"]

    df_.loc[:, "r_eval_practicas_aprob"] = df_["n_eval_practicas_aprob"] / df_["n_eval_practicas"]
    df_.loc[:, "r_eval_examenes_aprob"] = df_["n_eval_examenes_aprob"] / df_["n_eval_examenes"]
    df_.loc[:, "r_eval_trabajos_aprob"] = df_["n_eval_trabajos_aprob"] / df_["n_eval_trabajos"]
    df_.loc[:, "r_eval_participacion_aprob"] = df_["n_eval_participacion_aprob"] / df_["n_eval_participacion"]

    df_.loc[:, "avg_evaluaciones"] = df_["sum_notas_eval_total"] / df_["n_eval"]
    df_.loc[:, "avg_evaluaciones_aprob"] = df_["sum_notas_eval_aprob"] / df_["n_eval_aprob"]

    df_.loc[:, "avg_practicas"] = df_["sum_notas_eval_practicas"] / df_["n_eval_practicas"]
    df_.loc[:, "avg_examenes"] = df_["sum_notas_eval_examenes"] / df_["n_eval_examenes"]
    df_.loc[:, "avg_trabajos"] = df_["sum_notas_eval_trabajos"] / df_["n_eval_trabajos"]
    df_.loc[:, "avg_participacion"] = df_["sum_notas_eval_participacion"] / df_["n_eval_participacion"]

    df_.loc[:, "avg_practicas_aprob"] = df_["sum_notas_eval_practicas_aprob"] / df_["n_eval_practicas_aprob"]
    df_.loc[:, "avg_examenes_aprob"] = df_["sum_notas_eval_examenes_aprob"] / df_["n_eval_examenes_aprob"]
    df_.loc[:, "avg_trabajos_aprob"] = df_["sum_notas_eval_trabajos_aprob"] / df_["n_eval_trabajos_aprob"]
    df_.loc[:, "avg_participacion_aprob"] = df_["sum_notas_eval_participacion_aprob"] / df_["n_eval_participacion_aprob"]
    

    if features_dict is not None:
        df_ = df_[features_dict.keys()]

    return df_


def processing_file_ac_progress(df: pd.DataFrame, features_list: list) -> pd.DataFrame:
    
    # Crear una copia del DataFrame original para evitar modificar los datos originales
    df_ = df.copy()

    # Ordenar el DataFrame por múltiples columnas para asegurar un orden coherente en los datos
    df_ = df_.sort_values(["periodo", "cod_alumno", "cod_curso", "fecha_creacion", "cod_tipo_prueba", "num_prueba"], ignore_index=True)

    # Combinar el código de tipo de prueba con el número de prueba para crear un identificador único para cada tipo de prueba
    df_["cod_tipo_prueba_num"] = df_["cod_tipo_prueba"] + df_["num_prueba"].astype(str)
    
    # Reemplazar las notas con valor 88 por 0, asumiendo que 88 representa una condición especial o error en la entrada de datos
    df_["nota"] = df_["nota"].replace(88, 0)

    # Filtrar el DataFrame para incluir solo las notas válidas dentro del rango aceptable y excluir casos específicos
    df_ = df_.loc[
        (df_["nota"] >= 0)
        & (df_["nota"] <= 20)
        & (df_["nota"] != 55) # Excluir notas con valor 55, indicando que el alumno se retiró del curso
        & (df_["cod_tipo_prueba"] != "PF") # Excluir pruebas finales (PF) del análisis
    ]

    if not df.empty:
        # Aplicar una función para ajustar las fórmulas de evaluación a un formato estándar
        df_["eval_formula"] = df_["formula"].apply(ajustar_formula)

        # Agrupar por varios criterios y transformar los datos a un diccionario donde las claves son los tipos de prueba y los valores son las notas
        df_notas = (
            df_.groupby(
                ["periodo", "cod_alumno", "fecha_corte", "cod_curso", "eval_formula"],
                dropna=False
            )
            .apply(lambda row: dict(zip(row["cod_tipo_prueba_num"], row["nota"])))
            .reset_index(name="notas")
        )

        # Extraer los pesos de las evaluaciones desde las fórmulas ajustadas
        df_notas["pesos"] = df_notas["eval_formula"].apply(extraer_pesos)

        # Calcula el avance del curso basado en los pesos de las evaluaciones rendidas
        df_notas["pct_avance"] = df_notas.apply(avance_curso, axis=1)

        # Flag que indica si el alumno termino por completo el curso
        df_notas["flag_curso_completado"] = (df_notas["pct_avance"] == 100).astype(int)

        # Calcula el promedio final del curso basado en las notas y pesos de las evaluaciones rendidas
        df_notas["prom_final_actual"] = df_notas.apply(calculo_posible_prom_final, axis=1)

        # Calcula la posible nota faltante si en el mejor de los casos el alumno saca 20 en sus evaluaciones restantes
        df_notas["prom_max_restante"] = df_notas.apply(calcular_nota_faltante_mc, axis=1)

        # Calcula el posible promedio final en el mejor de los casos usando la columna prom_max_restante
        df_notas["prom_final_max_posible"] = df_notas["prom_final_actual"] + df_notas["prom_max_restante"]

        # Calcula el promedio ponderado actual del curso basado en las notas y pesos de las evaluaciones rendidas
        # df_notas["prom_ponderado_actual"] = df_notas.groupby(["periodo", "cod_alumno"])["prom_final_actual"].transform(lambda x: round(np.average(x, weights=df_notas.loc[x.index, "cant_creditos"]), 2))

        # Seleccionar y retornar solo las columnas relevantes para el análisis final
        df_notas = df_notas[features_list]

        return df_notas
    
    return pd.DataFrame(columns=features_list)

def ajustar_formula(cadena):
    """ Ajusta los porcentajes en las fórmulas de evaluación, reemplazando y normalizando su notación. """
    def reemplazo_prom(match):
        porcentaje = float(match.group(1))
        argumentos = match.group(3).split(",")
        evaluacion = argumentos[0]
        cant_evaluaciones = int(argumentos[1])
        nuevo_valor = round(porcentaje / cant_evaluaciones, 2)
        return " + ".join(
            [
                f"{nuevo_valor}*{evaluacion}{numero_evaluacion}"
                for numero_evaluacion in range(1, cant_evaluaciones + 1)
            ]
        )

    def reemplazo_normal(match):
        numero = match.group(1)
        argumento = match.group(3)
        return f"{numero}*{argumento}"

    cadena = re.sub(r"(\d+(\.\d+)?)% PROM\(([^,)]+(?:,[^,)]+)*)\)", reemplazo_prom, cadena)
    cadena = re.sub(r"(\d+(\.\d+)?)% \(([^)]+)\)", reemplazo_normal, cadena)
    return cadena


def extraer_pesos(formula):
    """ Extrae los pesos asignados a cada componente de la evaluación desde la fórmula proporcionada. """
    return {item.split("*")[1].strip(): float(item.split("*")[0]) for item in formula.split("+")}


def avance_curso(row):
    """ Determina el avance del curso basado en las evaluaciones rendidas y sus respectivos pesos. """
    notas, pesos = row["notas"], row["pesos"]
    evaluaciones_rendidas = notas.keys() & pesos.keys()
    avance = round(sum(pesos[key] for key in evaluaciones_rendidas), 2)
    return avance if avance < 99 else 100


def calculo_posible_prom_final(row):
    """ Calcula el promedio ponderado final utilizando las notas y sus correspondientes pesos. """
    notas, pesos = row["notas"], row["pesos"]
    total = sum(notas[key] * pesos[key] for key in notas if key in pesos)
    return round(total / 100, 2)


def calcular_nota_faltante_mc(row):
    """Calcula la nota faltante asumiendo que el alumno saque 20 en el mejor de los casos"""
    peso = row["pesos"]
    notas = row["notas"]
    claves_sin_notas = set(peso.keys()) - set(notas.keys())
    nota_faltante = round(sum([peso[clave] * 20 for clave in claves_sin_notas]) / 100, 2)
    return nota_faltante
