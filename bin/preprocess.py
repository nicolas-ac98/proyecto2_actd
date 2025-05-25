import pandas as pd
from sklearn.preprocessing import StandardScaler
import unidecode


def process():
    # Dataset ---------------------------------------------------------
    df_origin = pd.read_csv("../data/Base de Datos - Icfes.zip")
    print("Carga correcta de los datos")

    df = df_origin.copy()
    
    # Tranformacion de variables
    # standarize_number_variables(df)
    standarize_categorical_variables(df)
    print("Estandarización de variables")

    # Remover variables innecesarias
    drop_variables(df)
    print("Eliminacion de variables")

    # Remove duplicates
    df = df.drop_duplicates()
    # Guardar archivo
    print("Guardar archivo")
    # df.to_csv("../data/dataset_v1.csv.zip",index=False)

    get_values_for_app(df_origin=df_origin, df=df)

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
        , 'PUNT_INGLES'
        , 'PUNT_MATEMATICAS'
        , 'PUNT_SOCIALES_CIUDADANAS'
        , 'PUNT_C_NATURALES'
        , 'PUNT_LECTURA_CRITICA'
        , 'PUNT_GLOBAL'
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

def standarize_categorical_variables(df):
    # Seleccionar columnas categóricas (strings)
    cat_cols = list(df.select_dtypes(include='object').columns)
    # Aplicar la limpieza
    for col in cat_cols:
        df[col] = df[col].apply(limpiar_texto)
    
    # Transformar variables a dummy 
    # cat_cols.remove(interest_value)
    # df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # Transformar variables binarias
    standarize_binary(df=df,
                      variable='COLE_AREA_UBICACION', new_variable='COLE_AREA_URBANO', 
                      true_condition='urbano')

    standarize_binary(df=df,
                      variable='COLE_BILINGUE', new_variable='BILINGUE', 
                      true_condition='s')
    
    standarize_binary(df=df,
                      variable='COLE_CALENDARIO', new_variable='CALEN_A', 
                      true_condition='a')
    
    standarize_binary(df=df,
                      variable='COLE_NATURALEZA', new_variable='COLE_OFICIAL', 
                      true_condition='oficial')
    
    standarize_binary(df=df,
                      variable='COLE_SEDE_PRINCIPAL', new_variable='SEDE_PRINCIPAL', 
                      true_condition='s')
    
    standarize_binary(df=df,
                      variable='ESTU_GENERO', new_variable='SEXO_FEM', 
                      true_condition='f')
    
    standarize_binary(df=df,
                      variable='FAMI_TIENEAUTOMOVIL', new_variable='TIENE_AUTOMOVIL', 
                      true_condition='si')
    
    standarize_binary(df=df,
                      variable='FAMI_TIENECOMPUTADOR', new_variable='TIENE_COMPUTADOR', 
                      true_condition='si')
    
    standarize_binary(df=df,
                      variable='FAMI_TIENEINTERNET', new_variable='TIENE_INTERNET', 
                      true_condition='si')
    
    standarize_binary(df=df,
                      variable='FAMI_TIENELAVADORA', new_variable='TIENE_LAVADORA', 
                      true_condition='si')
    
    df['ESTRATOVIVIENDA'] = df['FAMI_ESTRATOVIVIENDA'].map({
            'estrato 1': 1,
            'estrato 2': 2,
            'estrato 3': 3,
            'estrato 4': 4,
            'estrato 5': 5,
            'estrato 6': 6
        }).fillna(0)
    
    df.drop(['FAMI_ESTRATOVIVIENDA'], axis=1, inplace=True)
    
    df['CUARTOSHOGAR'] = df['FAMI_CUARTOSHOGAR'].map({
            'uno': 1,
            'dos': 2,
            'tres': 3,
            'cuatro': 4,
            'cinco': 5,
            'seis': 6,
            'siete': 7,
            'ocho': 8,
            'nueve': 9,
            'diez_o_mas': 10
        }).fillna(0)
    
    df.drop(['FAMI_CUARTOSHOGAR'], axis=1, inplace=True)

    df['PERSONASHOGAR'] = df['FAMI_PERSONASHOGAR'].map({
            'una': 1,
            'dos': 2,
            'tres': 3,
            'cuatro': 4,
            'cinco': 5,
            'seis': 6,
            'siete': 7,
            'ocho': 8,
            'nueve': 9,
            'diez': 10,
            'once': 11,
            'doce_o_mas': 12
        }).fillna(0)
    
    df.drop(['FAMI_PERSONASHOGAR'], axis=1, inplace=True)

    df['DESEMP_INGLES'] = df['DESEMP_INGLES'].map({
            'a-': 0,
            'a1': 1,
            'a2': 2,
            'b1': 3,
            'b+': 4,
        })

def standarize_binary(df, variable, new_variable, true_condition):
    df[new_variable] = (df[variable] == true_condition).astype(int)
    df.drop([variable], axis=1, inplace=True)


def get_values_for_app(df_origin, df):

    df = df.add_prefix('preprocess_')

    df_merged = pd.concat([df_origin, df], axis=1)

    # print(df_merged.columns)

    cat_cols = [
        'COLE_CARACTER', 'COLE_DEPTO_UBICACION', 'COLE_GENERO', 'COLE_JORNADA',
        'ESTU_ESTADOINVESTIGACION', 'ESTU_NACIONALIDAD', 'FAMI_EDUCACIONMADRE',
        'FAMI_EDUCACIONPADRE'
        ]

    num_cols = ['ESTRATOVIVIENDA', 'CUARTOSHOGAR', 'PERSONASHOGAR']

    bin_cols = [
        'COLE_AREA_URBANO', 'BILINGUE', 'CALEN_A', 'COLE_OFICIAL', 
        'SEDE_PRINCIPAL', 'SEXO_FEM','TIENE_AUTOMOVIL', 'TIENE_COMPUTADOR', 
        'TIENE_INTERNET','TIENE_LAVADORA'
        ]
    
    for category in cat_cols:
        format_labels(df_merged,category)


def format_labels(df, variable):
    preprocess_variable = 'preprocess_' + variable
    unique_values = df[[variable,preprocess_variable]].drop_duplicates()
    
    unique_values = unique_values.rename(columns={variable: 'label', preprocess_variable: 'value'})

    # Convertir a lista de diccionarios
    unique_values.to_json("../data/option_"+ str.lower(variable), orient="records", force_ascii=False)

    # print(unique_values)

if __name__ == '__main__':
    process()