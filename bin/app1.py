import dash
from dash import dcc, html, Input, Output, State, ctx
import pandas as pd
import plotly.express as px
import requests
import json
import os
from model_load import new_estimation
from dash import callback_context
import mlflow


# Carga de datos
df = pd.read_csv("C:/Users/JESUS/OneDrive - Universidad de los andes/Maestría Ing Industrial/Analítica Computacional/Proyecto_2/Base de Datos - Icfes/Base2.csv")
niveles_orden = ['A-', 'A1', 'A2', 'B1', 'B+']
print('1')
# Ajuste de nombres con el mapa
map_nombres = {
    'VALLE': 'VALLE DEL CAUCA',
    'BOGOTÁ': 'SANTAFE DE BOGOTA D.C',
    'NORTE SANTANDER': 'NORTE DE SANTANDER',
    'SAN ANDRES': 'ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA'
}


with open("../data/option_cole_caracter", encoding="utf-8") as f:
    option_cole_caracter = json.load(f)

with open("../data/option_cole_depto_ubicacion", encoding="utf-8") as f:
    option_cole_depto_ubicacion = json.load(f)

with open("../data/option_cole_genero", encoding="utf-8") as f:
    option_cole_genero = json.load(f)

with open("../data/option_estu_estadoinvestigacion", encoding="utf-8") as f:
    option_estu_estadoinvestigacion = json.load(f)

with open("../data/option_estu_nacionalidad", encoding="utf-8") as f:
    option_estu_nacionalidad = json.load(f)
print('2')

nombres_amigables = {
    'COLE_AREA_URBANO': 'Ubicación Urbana',
    'BILINGUE': '¿Colegio Bilingüe?',
    'CALEN_A': 'Colegio con Calendario A',
    'SEDE_PRINCIPAL': '¿Es Sede Principal el Colegio?',
    'SEXO_FEM': 'Sexo',
    'TIENE_LAVADORA': '¿Tiene Lavadora?',
    'ESTRATOVIVIENDA': 'Estrato de Vivienda',
    'CUARTOSHOGAR': 'Cantidad de Cuartos',
    'ESTU_NACIONALIDAD': 'Nacionalidad',
    'COLE_CARACTER': 'Carácter del Colegio',
    'COLE_DEPTO_UBICACION': 'Departamento del Colegio',
    'COLE_GENERO': 'Tipo del Colegio por genero',
    'TIENE_AUTOMOVIL': 'Tiene Automóvil',
    'ESTU_ESTADOINVESTIGACION': 'Estado de Investigación'
}

opciones_categoricas = {
    'COLE_CARACTER': option_cole_caracter,
    'COLE_DEPTO_UBICACION': option_cole_depto_ubicacion,
    'COLE_GENERO': option_cole_genero,
    'ESTU_ESTADOINVESTIGACION': option_estu_estadoinvestigacion,
    'ESTU_NACIONALIDAD': option_estu_nacionalidad
}
# Cargar GeoJSON
geo_url = "https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/3aadedf47badbdac823b00dbe259f6bc6d9e1899/colombia.geo.json"
geojson_colombia = requests.get(geo_url).json()

ciudades = sorted(df['ESTU_MCPIO_RESIDE'].dropna().unique())
niveles = ['A-', 'A1', 'A2', 'B1', 'B+']

# Simulación de valores adicionales y sus tipos
defaults = {
    'COLE_AREA_URBANO': 1, 'BILINGUE': 0, 'CALEN_A': 1, 'SEDE_PRINCIPAL': 1,
    'SEXO_FEM': 1, 'TIENE_LAVADORA': 1, 'ESTRATOVIVIENDA': 1, 'CUARTOSHOGAR': 4, 'ESTU_NACIONALIDAD': 'colombia',
    'COLE_CARACTER': 'academico', 'COLE_DEPTO_UBICACION': 'bogota', 'COLE_GENERO': 'mixto',
    'TIENE_AUTOMOVIL': 0, 'ESTU_ESTADOINVESTIGACION': 'publicar'

}
tipo_variable = {
    'COLE_AREA_URBANO': 'binaria', 'BILINGUE': 'binaria', 'CALEN_A': 'binaria', 'SEDE_PRINCIPAL': 'binaria',
    'SEXO_FEM': 'binaria', 'TIENE_LAVADORA': 'binaria', 'ESTRATOVIVIENDA': 'numerica', 'CUARTOSHOGAR': 'numerica', 'ESTU_NACIONALIDAD': 'categorica',
    'COLE_CARACTER': 'categorica', 'COLE_DEPTO_UBICACION': 'categorica', 'COLE_GENERO': 'categorica',
    'TIENE_AUTOMOVIL': 'binaria'
}
print('3')
# APP
app = dash.Dash(__name__)
app.title = "Tablero ICFES - Nivel de Inglés"

app.layout = html.Div([
    html.Div([
        html.H1("INSTITUTO NACIONAL DE APRENDIZAJE", style={
            'textAlign': 'center',
            'color': 'white',
            'marginBottom': '5px',
            'fontFamily': 'Segoe UI, sans-serif'
        }),
        html.P(
            "Este tablero interactivo utiliza datos reales de resultados del examen ICFES para estimar el nivel de inglés de los estudiantes, "
            "basándose en características demográficas, académicas y del hogar.",
            style={
                'textAlign': 'center',
                'color': 'white',
                'fontSize': '16px',
                'marginTop': '10px',
                'marginBottom': '5px',
                'fontFamily': 'Segoe UI, sans-serif'
            }
        ),
        html.P(
            "El objetivo de este análisis es identificar patrones relevantes para apoyar estrategias de acción para las actividades del INA.",
            style={
                'textAlign': 'center',
                'color': 'white',
                'fontSize': '16px',
                'marginTop': '0px',
                'fontFamily': 'Segoe UI, sans-serif'
            }
        )
    ], style={
        'backgroundColor': '#004b6b',
        'padding': '20px',
        'borderBottom': '4px solid #013244'
    }),

    html.H2("Recolección de Datos para Predicción", style={'marginTop': '40px'}),

    html.H4("Preguntas obligatorias"),
    html.Div([
        html.Div([
            html.Label("Cuenta con Internet:"),
            dcc.Dropdown(id='input-internet', options=[
                    {"label": "Sí", "value": 1},
                    {"label": "No", "value": 0}
                ],
                placeholder='Cuenta con Internet',
                style={'width': '100%'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.Label("Colegio Oficial:"),
            dcc.Dropdown(id='input-oficial', options=[
                    {"label": "Sí", "value": 1},
                    {"label": "No", "value": 0}
                ],
                placeholder='Es Colegio Oficial',
                style={'width': '100%'})
        ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
        html.Div([
            html.Label("Tiene computador:"),
            dcc.Dropdown(id='input-pc', options=[
                    {"label": "Sí", "value": 1},
                    {"label": "No", "value": 0}
                ],
                placeholder='Tiene Computador',
                style={'width': '100%'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.Label("Jornada:"),
            dcc.Dropdown(id='input-jornada', options=[
                    {
                        "label": "MAÑANA",
                        "value": "manana"
                    },
                    {
                        "label": "COMPLETA",
                        "value": "completa"
                    },
                    {
                        "label": "SABATINA",
                        "value": "sabatina"
                    },
                    {
                        "label": "NOCHE",
                        "value": "noche"
                    },
                    {
                        "label": "TARDE",
                        "value": "tarde"
                    },
                    {
                        "label": "UNICA",
                        "value": "unica"
                    }
                ],
                placeholder='Jornada que Cursaba',
                style={'width': '100%'})
        ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
        html.Div([
            html.Label("Educación de la madre:"),
            dcc.Dropdown(id='input-edu-madre',options=[
                    {
                        "label": "Técnica o tecnológica completa",
                        "value": "tecnica_o_tecnologica_completa"
                    },
                    {
                        "label": "Educación profesional completa",
                        "value": "educacion_profesional_completa"
                    },
                    {
                        "label": "Secundaria (Bachillerato) completa",
                        "value": "secundaria_(bachillerato)_completa"
                    },
                    {
                        "label": "Primaria incompleta",
                        "value": "primaria_incompleta"
                    },
                    {
                        "label": "Postgrado",
                        "value": "postgrado"
                    },
                    {
                        "label": "No sabe",
                        "value": "no_sabe"
                    },
                    {
                        "label": "Ninguno",
                        "value": "ninguno"
                    },
                    {
                        "label": "Primaria completa",
                        "value": "primaria_completa"
                    },
                    {
                        "label": "Secundaria (Bachillerato) incompleta",
                        "value": "secundaria_(bachillerato)_incompleta"
                    },
                    {
                        "label": "Técnica o tecnológica incompleta",
                        "value": "tecnica_o_tecnologica_incompleta"
                    },
                    {
                        "label": "Educación profesional incompleta",
                        "value": "educacion_profesional_incompleta"
                    }
                ],
                placeholder='Educación de la madre',
                style={'width': '100%'}
            )], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.Label("Educación del padre:"),
            dcc.Dropdown(id='input-edu-padre', 
            options=[
                {
                    "label": "Técnica o tecnológica completa",
                    "value": "tecnica_o_tecnologica_completa"
                },
                {
                    "label": "Educación profesional completa",
                    "value": "educacion_profesional_completa"
                },
                {
                    "label": "Secundaria (Bachillerato) completa",
                    "value": "secundaria_(bachillerato)_completa"
                },
                {
                    "label": "Primaria incompleta",
                    "value": "primaria_incompleta"
                },
                {
                    "label": "Postgrado",
                    "value": "postgrado"
                },
                {
                    "label": "No sabe",
                    "value": "no_sabe"
                },
                {
                    "label": "Ninguno",
                    "value": "ninguno"
                },
                {
                    "label": "Primaria completa",
                    "value": "primaria_completa"
                },
                {
                    "label": "Secundaria (Bachillerato) incompleta",
                    "value": "secundaria_(bachillerato)_incompleta"
                },
                {
                    "label": "Técnica o tecnológica incompleta",
                    "value": "tecnica_o_tecnologica_incompleta"
                },
                {
                    "label": "Educación profesional incompleta",
                    "value": "educacion_profesional_incompleta"
                }
                ],
                placeholder='Educación de la padre',
                style={'width': '100%'}
            )], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
        html.Div([
            html.Label("Personas en el hogar:"),
            dcc.Dropdown(id='input-personas-hogar', options=[{'label': str(i), 'value': i} for i in range(1, 21)], placeholder='Seleccione una cantidad', style={'width': '100%'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'marginBottom': '30px'}),

    html.H4("Variables adicionales"),
    html.Div([
        dcc.Dropdown(
            id='selector-adicionales',
            options=[{'label': nombres_amigables[v], 'value': v} for v in tipo_variable.keys()],
            placeholder='Selecciona variable adicional a agregar'
        ),
        html.Div(id='contenedor-campos'),
        html.Div(id='resumen-adicionales', style={'marginTop': '20px'})
    ]),

    html.Div([
    html.Button("Predecir Nivel de Inglés", id='btn-predecir', n_clicks=0),
    html.Pre(id='output-variables', style={
        'whiteSpace': 'pre-wrap',
        'marginTop': '20px',
        'border': '1px solid #ccc',
        'padding': '10px',
        'backgroundColor': '#f9f9f9'
    })
    ]),


    html.Hr(),

    html.H2("Análisis Exploratorio", style={'fontFamily': 'Segoe UI, sans-serif'}),

    html.Div([
        html.Div([
            html.Label("Selecciona el nivel de inglés:",style={'fontFamily': 'Segoe UI, sans-serif'}),
            html.P("Visualiza la distribución de estudiantes por departamento para un nivel específico de inglés.",
               style={'fontFamily': 'Segoe UI, sans-serif', 'fontSize': '14px'}),
            dcc.Dropdown(id='nivel-dropdown', options=[{'label': n, 'value': n} for n in niveles], value='B1'),
            dcc.Graph(id='mapa-departamento')
        ], style={'width': '100%', 'marginTop': '30px','fontFamily': 'Segoe UI, sans-serif'}),

        html.Div([
            html.Div([
                html.Label("Selecciona la ciudad:",style={'fontFamily': 'Segoe UI, sans-serif'}),
                html.P("Compara los niveles de inglés alcanzados por los estudiantes en la ciudad seleccionada.",
                   style={'fontFamily': 'Segoe UI, sans-serif', 'fontSize': '14px'}),
                dcc.Dropdown(id='ciudad-dropdown', options=[{'label': c, 'value': c} for c in ciudades], value='MEDELLÍN',style={'fontFamily': 'Segoe UI, sans-serif'}),
                dcc.Graph(id='grafico-niveles')
            ], style={'width': '48%', 'marginRight': '2%'}),

            html.Div([
                html.Label("Selecciona ciudad y nivel para el Top 5 colegios:",style={'fontFamily': 'Segoe UI, sans-serif'}),
                html.P("Consulta los cinco colegios con mayor número de estudiantes en el nivel seleccionado dentro de la ciudad.",
                    style={'fontFamily': 'Segoe UI, sans-serif', 'fontSize': '14px'}),
                dcc.Dropdown(id='ciudad-colegio-dropdown', options=[{'label': c, 'value': c} for c in ciudades], value='MEDELLÍN',style={'fontFamily': 'Segoe UI, sans-serif'}),
                dcc.Dropdown(id='nivel-colegio-dropdown', options=[{'label': n, 'value': n} for n in niveles], value='B1',style={'fontFamily': 'Segoe UI, sans-serif'}),
                dcc.Graph(id='grafico-colegios')
            ], style={'width': '48%'})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '30px','fontFamily': 'Segoe UI, sans-serif'})
    ])
])

#CALLBACKS
print('4')
#Mapa por departamento
@app.callback(
    Output('mapa-departamento', 'figure'),
    Input('nivel-dropdown', 'value')
)
def actualizar_mapa(nivel):
    df_filtrado = df[df['DESEMP_INGLES'] == nivel]
    df_dep = df_filtrado.groupby('ESTU_DEPTO_RESIDE').size().reset_index(name='Cantidad')
    df_dep.rename(columns={'ESTU_DEPTO_RESIDE': 'Departamento'}, inplace=True)
    df_dep['Departamento'] = df_dep['Departamento'].replace(map_nombres)

    todos_dptos = [f['properties']['NOMBRE_DPT'] for f in geojson_colombia['features']]

    df_base = pd.DataFrame({'Departamento': todos_dptos})
    df_dep = df_base.merge(df_dep, on='Departamento', how='left')
    df_dep['Cantidad'] = df_dep['Cantidad'].fillna(0)

    fig = px.choropleth(
        df_dep,
        geojson=geojson_colombia,
        featureidkey="properties.NOMBRE_DPT",
        locations="Departamento",
        color="Cantidad",
        color_continuous_scale="Blues",
        title=f"Estudiantes con nivel {nivel} por departamento"
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r":0, "t":40, "l":0, "b":0})
    return fig

#Gráfico de niveles por ciudad
@app.callback(
    Output('grafico-niveles', 'figure'),
    Input('ciudad-dropdown', 'value')
)
def grafico_niveles_ciudad(ciudad):
    df_ciudad = df[df['ESTU_MCPIO_RESIDE'] == ciudad]
    conteo = df_ciudad['DESEMP_INGLES'].value_counts().reindex(niveles_orden, fill_value=0).reset_index()
    conteo.columns = ['Nivel', 'Cantidad']
    colores = {
        'A-': '#d73027', 'A1': '#fc8d59', 'A2': '#fee090', 'B1': '#91bfdb', 'B+': '#4575b4'
    }

    fig = px.bar(
        conteo,
        x='Cantidad',
        y='Nivel',
        orientation='h',
        color='Nivel',
        color_discrete_map=colores,
        title=f"Niveles de inglés en {ciudad}"
    )
    fig.update_traces(marker_line_color='black', marker_line_width=1.2)
    fig.update_layout(
        yaxis=dict(categoryorder='array', categoryarray=niveles_orden, title_font=dict(size=16)),
        xaxis=dict(title_font=dict(size=16)),
        title_font_size=20,
        plot_bgcolor='white'
    )
    return fig

#Top 5 colegios por ciudad y nivel
@app.callback(
    Output('grafico-colegios', 'figure'),
    [Input('ciudad-colegio-dropdown', 'value'),
     Input('nivel-colegio-dropdown', 'value')]
)
def grafico_top_colegios(ciudad, nivel):
    df_top = df[
        (df['ESTU_MCPIO_RESIDE'] == ciudad) &
        (df['DESEMP_INGLES'] == nivel)
    ]
    df_colegios = df_top['COLE_NOMBRE_SEDE'].value_counts().reset_index()
    df_colegios.columns = ['Colegio', 'Estudiantes']
    df_top5 = df_colegios.head(5)
    colores = ['#4575b4', '#91bfdb', '#fee090', '#fc8d59', '#d73027']

    fig = px.bar(
        df_top5,
        x='Colegio',
        y='Estudiantes',
        color='Colegio',
        color_discrete_sequence=colores,
        title=f"Top 5 colegios con nivel {nivel} en {ciudad}"
    )
    fig.update_traces(marker_line_color='black', marker_line_width=1.2)
    fig.update_layout(
        xaxis_title='Colegio',
        yaxis_title='Estudiantes',
        title_font_size=20,
        showlegend=False,
        plot_bgcolor='white'
    )
    return fig


@app.callback(
    Output('contenedor-campos', 'children'),
    Output('resumen-adicionales', 'children'),
    Input('selector-adicionales', 'value'),
    State('contenedor-campos', 'children'),
    State('resumen-adicionales', 'children'),
    State({'type': 'dynamic-input', 'index': dash.dependencies.ALL}, 'value'),
    State({'type': 'dynamic-input', 'index': dash.dependencies.ALL}, 'id'),
    prevent_initial_call=True
)
def agregar_y_actualizar(variable, campos_actuales, resumen_actual, valores, ids):
    if campos_actuales is None:
        campos_actuales = []
    if resumen_actual is None:
        resumen_actual = []

    # Crear nuevo campo si no existe
    campos_ids = [c['props']['id'] if isinstance(c, dict) else getattr(c, 'id', None) for c in campos_actuales]
    if variable and f'adicional-{variable}' not in campos_ids:
        
        if tipo_variable[variable] == 'binaria':
            if variable == 'SEXO_FEM':
                opciones = [{'label': 'Femenino', 'value': 1}, {'label': 'Masculino', 'value': 0}]
            else:
                opciones = [{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}]

            nuevo_input = dcc.Dropdown(
                id={'type': 'dynamic-input', 'index': variable},
                options=opciones,
                placeholder=nombres_amigables.get(variable, variable.replace('_', ' ').title()),
                style={'width': '100%'}
            )

        elif tipo_variable[variable] == 'numerica':
            nuevo_input = dcc.Input(
                id={'type': 'dynamic-input', 'index': variable},
                type='number',
                placeholder=variable.replace('_', ' ').title(),
                style={'width': '100%'}
            )
        elif tipo_variable[variable] == 'categorica':
            nuevo_input = dcc.Dropdown(
                id={'type': 'dynamic-input', 'index': variable},
                options=opciones_categoricas.get(variable, []),
                placeholder=variable.replace('_', ' ').title(),
                style={'width': '100%'}
            )
        else:
            nuevo_input = dcc.Input(
                id={'type': 'dynamic-input', 'index': variable},
                type='text',
                placeholder=variable.replace('_', ' ').title(),
                style={'width': '100%'}
            )

        grupo = html.Div([
            html.Label(nombres_amigables.get(variable, variable.replace('_', ' ').title()), style={'display': 'block'}),
            nuevo_input
        ], id=f'adicional-{variable}', style={
            'marginBottom': '10px',
            'width': '48%',
            'display': 'inline-block',
            'marginRight': '2%'
        })

        campos_actuales.append(grupo)

        resumen_actual.append(html.P(
            f"{variable.replace('_', ' ').title()}: Valor no especificado",
            id=f"resumen-{variable}"
        ))

    # Actualiza resumen según todos los valores actuales
    resumen_actualizado = []
    encontrados = set()
    for val, id_dict in zip(valores, ids):
        variable_key = id_dict['index']
        nombre = nombres_amigables.get(variable_key, variable_key.replace('_', ' ').title())
        texto = f"{nombre}: {val if val is not None else 'Valor no especificado'}"
        encontrados.add(variable_key)
        resumen_actualizado.append(html.P(texto, id=f"resumen-{variable_key}"))

    for resumen in resumen_actual:
        if hasattr(resumen, 'props') and 'id' in resumen.props:
            clave = resumen.props['id'].replace('resumen-', '')
            if clave not in encontrados:
                resumen_actualizado.append(resumen)

    return campos_actuales, resumen_actualizado


@app.callback(
    Output('output-variables', 'children'),
    Input('btn-predecir', 'n_clicks'),
    State('input-internet', 'value'),
    State('input-oficial', 'value'),
    State('input-pc', 'value'),
    State('input-jornada', 'value'),
    State('input-edu-madre', 'value'),
    State('input-edu-padre', 'value'),
    State('input-personas-hogar', 'value'),
    State({'type': 'dynamic-input', 'index': dash.dependencies.ALL}, 'value'),
    State({'type': 'dynamic-input', 'index': dash.dependencies.ALL}, 'id'),
    prevent_initial_call=True
)
def predecir_ingles(n_clicks, internet, oficial, pc, jornada, edu_madre, edu_padre, personas, adicionales_valores, adicionales_ids):
    try:
        if n_clicks == 0:
            return ""

        print("1 - Entró al callback")

        datos = {
            'TIENE_INTERNET': internet,
            'COLE_OFICIAL': oficial,
            'TIENE_COMPUTADOR': pc,
            'COLE_JORNADA': jornada,
            'FAMI_EDUCACIONMADRE': edu_madre,
            'FAMI_EDUCACIONPADRE': edu_padre,
            'PERSONASHOGAR': personas
        }

        print("2 - Obligatorios:", datos)

        #Convertir adicionales a diccionario temporal
        adicionales_dict = {id_['index']: val for val, id_ in zip(adicionales_valores, adicionales_ids)}

        for clave in defaults:
            datos[clave] = adicionales_dict.get(clave, defaults[clave])


        print("3 - Con adicionales:", datos)

        df = pd.DataFrame([datos])
        print("4 - DataFrame construido:", df)

        #Llamar la función de predicción
        nivel, prob = new_estimation(df)

        print("5 - Predicción realizada")

        return f"Nivel de inglés predicho: {nivel}\nConfianza: {prob:.2%}"

    except Exception as e:
        print("Error en callback:", e)
        return f"Error en predicción: {e}"


if __name__ == '__main__':
    app.run_server(debug=True)