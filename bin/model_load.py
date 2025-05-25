import mlflow
import joblib
import pandas as pd
import json
import numpy as np
from tensorflow import keras

def new_estimation(new_data, version_model = 'pmv_v3', preprocessor='preprocessors/english_performance_preprocessor_v1.pkl'):
    # Cargar el preprocesador
    preprocessor = joblib.load(preprocessor)

    # Obtener el experimento por nombre
    experiment = mlflow.get_experiment_by_name("English performance model")

    # Obtener todos los runs de ese experimento
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    # Filtrar por nombre del run
    run = runs[runs['tags.mlflow.runName'] == version_model].iloc[0]
    run_id = run.run_id

    # Cargar el modelo del run
    model = keras.models.load_model(f"modelo_v7.keras")

    # Aplicar el preprocesador
    new_data_processed = preprocessor.transform(new_data)

    # Obtener probabilidades por clase
    y_pred_probs = model.predict(new_data_processed)

    # Obtener clases predichas
    y_pred_class = np.argmax(y_pred_probs, axis=1)

    cat = ['A-','A1','A2','B1','B+']
    return cat[y_pred_class[0]], max(y_pred_probs[0])


if __name__ == '__main__':
    # Suponiendo que tienes un nuevo DataFrame con las mismas columnas que X
    # Leer el archivo
    with open("../data/input_request.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convertir a DataFrame
    new_data = pd.DataFrame([data])

    print(new_estimation(new_data))