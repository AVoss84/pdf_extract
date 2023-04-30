from sklearn.pipeline import Pipeline
from pdf_extract.services.file import PickleService 
from pdf_extract.config import global_config as glob

cleaner = PickleService(path="trained_text_preproc.pkl", root_path=glob.UC_DATA_DIR, is_df=False, verbose=False).doRead()
model = PickleService(path="trained_model.pkl", root_path=glob.UC_DATA_DIR, is_df=False, verbose=False).doRead()

pipe = Pipeline([('cleaner', cleaner), ('model', model)])
