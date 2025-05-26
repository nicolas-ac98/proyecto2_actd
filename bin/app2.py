import dash
from dash import html

# Crear app
app = dash.Dash(__name__)

# Layout mínimo
app.layout = html.Div([
    html.H1("Hola Mundo con Dash"),
    html.P("Si ves esto, tu app Dash está funcionando correctamente.")
])

# Ejecutar app
if __name__ == '__main__':
    app.run_server(debug=True)
