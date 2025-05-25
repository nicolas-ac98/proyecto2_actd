import dash
from dash import dcc, html, Input, Output, State, ctx
import pandas as pd
import plotly.express as px
import requests
import json

# Carga de datos
df = pd.read_csv("Base_de_Datos_Icfes.csv")
niveles_orden = ['A-', 'A1', 'A2', 'B1', 'B+']

# Ajuste de nombres con el mapa
map_nombres = {
    'VALLE': 'VALLE DEL CAUCA',
    'BOGOTÁ': 'SANTAFE DE BOGOTA D.C',
    'NORTE SANTANDER': 'NORTE DE SANTANDER',
    'SAN ANDRES': 'ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA'
}

# Cargar GeoJSON
geo_url = "https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/3aadedf47badbdac823b00dbe259f6bc6d9e1899/colombia.geo.json"
geojson_colombia = requests.get(geo_url).json()

ciudades = sorted(df['ESTU_MCPIO_RESIDE'].dropna().unique())
niveles = ['A-', 'A1', 'A2', 'B1', 'B+']

# Simulación de valores adicionales y sus tipos
defaults = {
    'UBICADO_URBANO': 'Sí', 'BILINGUE': 'No', 'CALENDARIO_A': 'Sí', 'SEDE_PRINCIPAL': 'Sí',
    'SEXO': 'Femenino', 'LAVADORA': 'Sí', 'ESTRATO': 3, 'CUARTOS_HOGAR': 4
}
tipo_variable = {
    'UBICADO_URBANO': 'binaria', 'BILINGUE': 'binaria', 'CALENDARIO_A': 'binaria', 'SEDE_PRINCIPAL': 'binaria',
    'SEXO': 'binaria', 'LAVADORA': 'binaria', 'ESTRATO': 'numerica', 'CUARTOS_HOGAR': 'numerica'
}

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
        html.P("Observación del estimado del nivel de inglés respecto a características demográficas y geográficas del estudiante.", style={
            'textAlign': 'center',
            'color': 'white',
            'fontSize': '16px',
            'marginTop': '0px',
            'fontFamily': 'Segoe UI, sans-serif'
        })
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
            dcc.Dropdown(id='input-internet', options=[{'label': i, 'value': i} for i in ['Sí', 'No']], style={'width': '100%'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.Label("Colegio Oficial:"),
            dcc.Dropdown(id='input-oficial', options=[{'label': i, 'value': i} for i in ['Sí', 'No']], style={'width': '100%'})
        ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
        html.Div([
            html.Label("Tiene computador:"),
            dcc.Dropdown(id='input-pc', options=[{'label': i, 'value': i} for i in ['Sí', 'No']], style={'width': '100%'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.Label("Jornada:"),
            dcc.Dropdown(id='input-jornada', options=[{'label': j, 'value': j} for j in ['Completa', 'Mañana', 'Tarde', 'Noche']], style={'width': '100%'})
        ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
        html.Div([
            html.Label("Educación de la madre:"),
            dcc.Dropdown(id='input-edu-madre', options=[{'label': e, 'value': e} for e in ['Ninguna', 'Básica', 'Media', 'Técnica', 'Superior']], style={'width': '100%'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.Label("Educación del padre:"),
            dcc.Dropdown(id='input-edu-padre', options=[{'label': e, 'value': e} for e in ['Ninguna', 'Básica', 'Media', 'Técnica', 'Superior']], style={'width': '100%'})
        ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
        html.Div([
            html.Label("Personas en el hogar:"),
            dcc.Dropdown(id='input-personas-hogar', options=[{'label': str(i), 'value': i} for i in range(1, 21)], placeholder='Seleccione una cantidad', style={'width': '100%'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'marginBottom': '30px'}),

    html.H4("Variables adicionales"),
    html.Div([
        dcc.Dropdown(
            id='selector-adicionales',
            options=[{'label': v.replace('_', ' ').title(), 'value': v} for v in tipo_variable.keys()],
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
            dcc.Dropdown(id='nivel-dropdown', options=[{'label': n, 'value': n} for n in niveles], value='B1'),
            dcc.Graph(id='mapa-departamento')
        ], style={'width': '100%', 'marginTop': '30px','fontFamily': 'Segoe UI, sans-serif'}),

        html.Div([
            html.Div([
                html.Label("Selecciona la ciudad:",style={'fontFamily': 'Segoe UI, sans-serif'}),
                dcc.Dropdown(id='ciudad-dropdown', options=[{'label': c, 'value': c} for c in ciudades], value='MEDELLÍN',style={'fontFamily': 'Segoe UI, sans-serif'}),
                dcc.Graph(id='grafico-niveles')
            ], style={'width': '48%', 'marginRight': '2%'}),

            html.Div([
                html.Label("Selecciona ciudad y nivel para el Top 5 colegios:",style={'fontFamily': 'Segoe UI, sans-serif'}),
                dcc.Dropdown(id='ciudad-colegio-dropdown', options=[{'label': c, 'value': c} for c in ciudades], value='MEDELLÍN',style={'fontFamily': 'Segoe UI, sans-serif'}),
                dcc.Dropdown(id='nivel-colegio-dropdown', options=[{'label': n, 'value': n} for n in niveles], value='B1',style={'fontFamily': 'Segoe UI, sans-serif'}),
                dcc.Graph(id='grafico-colegios')
            ], style={'width': '48%'})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '30px','fontFamily': 'Segoe UI, sans-serif'})
    ])
])

#CALLBACKS

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

    #Crear nuevo campo si no existe
    campos_ids = [c['props']['id'] if isinstance(c, dict) else getattr(c, 'id', None) for c in campos_actuales]
    if variable and f'adicional-{variable}' not in campos_ids:
        nuevo_input = dcc.Dropdown(
            id={'type': 'dynamic-input', 'index': variable},
            options=[{'label': 'Sí', 'value': 'Sí'}, {'label': 'No', 'value': 'No'}],
            placeholder=variable.replace('_', ' ').title(),
            style={'width': '100%'}
        ) if tipo_variable[variable] == 'binaria' else dcc.Input(
            id={'type': 'dynamic-input', 'index': variable},
            type='number', min=1, max=10,
            placeholder=variable.replace('_', ' ').title(),
            style={'width': '100%'}
        )

        grupo = html.Div([
            html.Label(variable.replace('_', ' ').title(), style={'display': 'block'}),
            nuevo_input
        ], id=f'adicional-{variable}', style={'marginBottom': '10px', 'width': '48%', 'display': 'inline-block', 'marginRight': '2%'})

        campos_actuales.append(grupo)

        #línea vacía al resumen por defecto
        resumen_actual.append(html.P(f"{variable.replace('_', ' ').title()}: Valor no especificado", id=f"resumen-{variable}"))

    #Actualiza resumen según todos los valores actuales
    resumen_actualizado = []
    encontrados = set()
    for val, id_dict in zip(valores, ids):
        variable_key = id_dict['index']
        nombre = variable_key.replace('_', ' ').title()
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
def recolectar_variables(n_clicks, internet, oficial, pc, jornada, edu_madre, edu_padre, personas, valores_adic, ids_adic):
    #Variables obligatorias
    data = {
        'INTERNET': internet,
        'COLEGIO_OFICIAL': oficial,
        'TIENE_PC': pc,
        'JORNADA': jornada,
        'EDU_MADRE': edu_madre,
        'EDU_PADRE': edu_padre,
        'PERSONAS_HOGAR': personas
    }

    #Variables adicionales
    adicionales = {i['index']: val if val is not None else defaults[i['index']] for val, i in zip(valores_adic, ids_adic)}
    for clave in defaults:
        if clave not in adicionales:
            adicionales[clave] = defaults[clave]
    data.update(adicionales)
    return json.dumps(data, indent=2, ensure_ascii=False)



if __name__ == '__main__':
    app.run_server(debug=True)