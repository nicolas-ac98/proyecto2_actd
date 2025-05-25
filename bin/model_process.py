# Librerias --------------------------------------------------------
# Importe MLflow, ketas y tensorflow
import pandas as pd
import mlflow 
import mlflow.keras
import keras
import tensorflow as tf
import argparse # Usaremos argparse para pasarle argumentos a las funciones de entrenamiento
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from mlflow.models.signature import infer_signature
import joblib
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np


def process(epochs=10, batch_size=32, learning_rate=0.001):
    # Dataset ---------------------------------------------------------
    # Obtenemos el dataset MNIST 
    df = pd.read_csv("../data/dataset_v1.csv.zip")

    # Dividir el conjunto en variable de interes y predictores
    interest_variable = 'DESEMP_INGLES'
    X = df.drop(interest_variable, axis=1)

    y_processed = tf.keras.utils.to_categorical(df[interest_variable], num_classes=5)
    print("Se cargo el conjunto de datos")
    print("Variable de interés {}".format(interest_variable))
    print("Num. variable predictoras: {}".format(len(X.columns.to_list())))
    print("Lista variable predictoras: {}".format(X.columns.to_list()))
    print("Num registros: {}".format(len(X)))

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

    # 2. Definir transformaciones
    preprocessor = ColumnTransformer([
        # One-Hot para categóricas
        ('ohe', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        # Escalado para numéricas
        ('scale', StandardScaler(), num_cols),
        # Passthrough para binarias
        ('pass', 'passthrough', bin_cols)
    ])

    # 3. Transformar datos
    X_processed = preprocessor.fit_transform(X)

    # Dividimos el conjunto de datos en train test
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, 
                                                        test_size=0.2, random_state=73)
    

    # 5. Construir el modelo
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64,  activation='relu'),
        tf.keras.layers.Dense(32,  activation='relu'),
        tf.keras.layers.Dense(5,  activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    experiment = mlflow.set_experiment("English performance model")
    
     # 6. Run de MLflow
    with mlflow.start_run(run_name="pmv_v4",
                          experiment_id=experiment.experiment_id):
        # Parámetros
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("categorical_cols", cat_cols)
        mlflow.log_param("numeric_cols", num_cols)
        mlflow.log_param("binary_cols", bin_cols)

        # Entrenamiento
        model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size)

        # Evaluación
        # accuracy
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("test_accuracy", accuracy)

        # accuracy
        # 1. Obtener predicciones de clases (no probabilidades)
        y_pred_probs = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # 2. Reporte de clasificación
        report = classification_report(y_true_classes, y_pred_classes, output_dict=True)

        # Log de métricas clave
        mlflow.log_metric("precision_macro", report["macro avg"]["precision"])
        mlflow.log_metric("recall_macro", report["macro avg"]["recall"])
        mlflow.log_metric("f1_macro", report["macro avg"]["f1-score"])

        for i in range(0,5):

            cat = ['a-less','a1','a2','b1','b-plus']

            mlflow.log_metric("precision_" + cat[i], report[str(i)]["precision"])
            mlflow.log_metric("recall_" + cat[i], report[str(i)]["recall"])
            mlflow.log_metric("f1_" + cat[i], report[str(i)]["f1-score"])

        # 3. ROC AUC para cada clase (multiclase)
        try:
            auc = roc_auc_score(y_test, y_pred_probs, multi_class="ovr")
            mlflow.log_metric("roc_auc_ovr", auc)
        except ValueError as e:
            print(f"ROC AUC no calculado: {e}")
            
        # Guardar modelo y preprocesador
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X[:5]
        input_example.to_csv("../data/input_example_v1.csv")
        mlflow.keras.log_model(
            model, 
            artifact_path="english_performance_model_v1",
            signature=signature
            # input_example=input_example
            )
        # Puedes también guardar el preprocessor como artefacto:
        joblib.dump(preprocessor, "preprocessors/english_performance_preprocessor_v1.pkl")
        mlflow.log_artifact("english_performance_preprocessor_v1.pkl")

        print(f"Test accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    process(epochs=30)