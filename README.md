# Template for generic Python package (Data Science) 

This is a blueprint of a generic end-to-end data science project, i.e. building a Python package along the usual steps: data preprocessing, model training, prediction, postprocessing, REST API construction (for real-time model serving) and containerization for final deployment as a microservice.

## Package structure

```
├── docker-compose.yaml
├── Dockerfile
├── logs
├── main.py                               # REST API definition 
├── README.md
├── requirements.txt
└── src
    ├── __init__.py
    ├── my_package
    │   ├── config
    │   │   ├── config.py
    │   │   ├── global_config.py          # user environemntal variables
    │   │   ├── __init__.py
    │   │   └── input_output.yaml         # structure reading and writing of files
    │   ├── data                          # temporary data dump (will be git ignored)
    │   ├── resources
    │   │   ├── __init__.py
    │   │   ├── postprocessor.py
    │   │   ├── predictor.py
    │   │   ├── preprocessor.py
    │   │   ├── README.md
    │   │   └── trainer.py
    │   ├── services
    │   │   ├── file.py
    │   │   ├── __init__.py
    │   │   ├── pipelines.py
    │   │   ├── publisher.py
    │   │   └── README.md
    │   └── utils
    │       ├── __init__.py
    │       └── utils.py
    ├── notebooks
    └── setup.py
```

## Use Case description

**Business goal**: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 

**Business stakeholders**: Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

**Input data description**: Iris data set

**Business impact KPI**: Faster STP (in hours/days)


## Package installation and application develoment

Create conda virtual environment with required packages 
```bash
conda create -n proj_templ python=3.8 -y
conda activate proj_templ
```

To install the package locally execute the following steps:

```bash
pip install -r requirements.txt         
pip install -e src                     # install own package
```

Start application locally:
```bash
uvicorn main:app --reload         # checkout Swagger docs: http://127.0.0.1:8000/docs 
```

Build image and run app in container:
```bash                                 
docker-compose up -d 
```

 
