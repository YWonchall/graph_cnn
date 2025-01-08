import types

def load_config(path):
    config = types.ModuleType("config")
    with open(path, 'r') as file:
        exec(file.read(), config.__dict__)
    # 访问配置
    return config
