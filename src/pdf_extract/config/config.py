from my_package.services.file import YAMLservice

my_yaml = YAMLservice(path = "my_package/config/input_output.yaml")
io = my_yaml.doRead()
