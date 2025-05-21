import pandas as pd
from sklearn.preprocessing import StandardScaler
import unidecode


def process():
    # Dataset ---------------------------------------------------------
    df = pd.read_csv("../data/Base de Datos - Icfes.zip")
    print("Carga correcta de los datos")
    
    # Remover variables innecesarias
    drop_variables(df)
    print("Eliminacion de variables")
    
    # Tranformacion de variables
    standarize_number_variables(df)
    df = standarize_categorical_variables(df=df, interest_value='DESEMP_INGLES')
    print("Estandarización de variables")
    # Remove duplicates
    df = df.drop_duplicates()
    # Guardar archivo
    print("guardar archivo")
    df.to_csv("../data/dataset_v1.csv.zip",index=False)

    print(df.columns)

# funciones
def drop_variables(df):
    variables_to_drop = [
        'PERIODO'
        , 'ESTU_CONSECUTIVO'
        , 'COLE_CODIGO_ICFES'
        , 'COLE_NOMBRE_ESTABLECIMIENTO'
        , 'ESTU_ESTUDIANTE'
        , 'COLE_COD_DANE_ESTABLECIMIENTO'
        , "COLE_COD_DANE_SEDE"
        , "COLE_COD_DEPTO_UBICACION"
        , "COLE_COD_MCPIO_UBICACION"
        , "COLE_CODIGO_ICFES"
        , "COLE_MCPIO_UBICACION"
        , "COLE_NOMBRE_ESTABLECIMIENTO"
        , "COLE_NOMBRE_SEDE"
        , "ESTU_COD_DEPTO_PRESENTACION"
        , "ESTU_COD_MCPIO_PRESENTACION"
        , "ESTU_COD_RESIDE_DEPTO"
        , "ESTU_COD_RESIDE_MCPIO"
        , "ESTU_DEPTO_PRESENTACION"
        , "ESTU_DEPTO_RESIDE"
        , "ESTU_MCPIO_PRESENTACION"
        , "ESTU_MCPIO_RESIDE"
        , "ESTU_PAIS_RESIDE"
        , "ESTU_PRIVADO_LIBERTAD"
        , "ESTU_FECHANACIMIENTO"
        , "ESTU_TIPODOCUMENTO"
    ]

    df.drop(variables_to_drop, axis=1, inplace=True)

# 
def standarize_number_variables(df):
    # Seleccionar solo columnas numéricas
    num_cols = df.select_dtypes(include='number').columns

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

# Función de limpieza de texto
def limpiar_texto(texto):
    if pd.isnull(texto):
        return "na"
    texto = unidecode.unidecode(texto)  # elimina tildes y reemplaza ñ por n
    texto = texto.replace(' ', '_')     # reemplaza espacios por guiones bajos
    return texto.lower()                # convierte a minúsculas (opcional)

def standarize_categorical_variables(df, interest_value):
    # Seleccionar columnas categóricas (strings)
    cat_cols = list(df.select_dtypes(include='object').columns)
    # Aplicar la limpieza
    for col in cat_cols:
        df[col] = df[col].apply(limpiar_texto)
    
    # Transformar variables a dummy 
    cat_cols.remove(interest_value)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    return df

if __name__ == '__main__':
    process()