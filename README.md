# A Seldon MLFLow Custom Loader

An extension of Seldons MLFlow prepackaged [model server](https://docs.seldon.io/projects/seldon-core/en/latest/servers/mlflow.html). 

If you are using
[Seldon](https://docs.seldon.io/projects/seldon-core/en/latest/index.html) to deploy your models and you would like to have the ability to run `predict_prob` on the underlying classifier, then you can might want to use
this extension. 

Seldon-core currently provides only support to load your MLFlow model as a pyfunc. Therefore, you can not run e.g. predict_prob.

# How to use:

Update you Seldon deployment with the custom model loader e.g. `MLFLOW_SERVER_CUSTOM`
.

    apiVersion: machinelearning.seldon.io/v1alpha2
    kind: SeldonDeployment
    metadata:
    name: mlflow
    spec:
    name: wines
    predictors:
        - graph:
            children: []
            implementation: MLFLOW_SERVER_CUSTOM
            modelUri: gs://seldon-models/mlflow/elasticnet_wine_1.8.0
            name: classifier
            parameters:
            - name: method
            type: STRING
            value: predict
        name: default
        replicas: 1

Make sure to make `MLFLOW_SERVER_CUSTOM` available in your [seldon-core-operator](https://docs.seldon.io/projects/seldon-core/en/latest/reference/helm.html) installation.

E.g

    predictor_servers:
        MLFLOW_SERVER_CUSTOM:
        protocols:
            seldon:
            defaultImageVersion: "1.0.0"
            image: hjilke/seldon-custom-mlflowserver


# Modify code and build the image:

In case you want to modify the code and build the image, you can simply run:

    make build
