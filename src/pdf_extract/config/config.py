from pdf_extract.services.file import YAMLservice

my_yaml = YAMLservice(path = "pdf_extract/config/input_output.yaml")
io = my_yaml.doRead()
