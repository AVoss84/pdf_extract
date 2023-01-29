# Text classification based on PDF input data

## Package structure

```
.
├── environment.yml
├── logs
├── main.py
├── README.md
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── notebooks
│   │   ├── fasttext_classifier.ipynb
│   │   └── naivebayes_classifier.ipynb
│   ├── pdf_extract
│   │   ├── config
│   │   ├── data
│   │   ├── resources
│   │   ├── services
│   │   └── utils
│   ├── setup.py
│   └── templates
└── stream_app.py
```


## Package installation

Create conda virtual environment with required packages 
```bash
conda env create -f environment.yml 
conda activate env_pdf
```

Install your package
```bash
python -m spacy download en_core_web_lg
python -m spacy download de_core_news_lg      # install large Glove engl. word embeddings
pip install -e src
``` 

Start REST API locally:
```bash
uvicorn main:app --reload --port 5000         # checkout Swagger docs: http://127.0.0.1:5000/docs 
``` 

Start streamlit app locally:
```bash
streamlit run stream_app.py     
``` 
