import yaml
import threading
import sys
import yaml
import argparse

def get_yaml_contents(filepath):
    with open(filepath, 'r', encoding='utf-8') as ymlfile:
        yaml_obj = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    return yaml_obj

class EasyDict(dict):
    """
    Convenience class that behaves exactly like dict(), but allows accessing
    the keys and values using the attribute syntax, i.e., "mydict.key = value".
    """
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]


class Configuration(EasyDict):
    """
    SingleObj Mode
    Configuration class for system config
    """
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs ):
        if not hasattr(Configuration, "_instance"):
            with Configuration._instance_lock:
                if not hasattr(Configuration, "_instance"):
                    Configuration._instance = EasyDict.__new__(cls)
                    Configuration._instance.init(*args, **kwargs)
        return Configuration._instance

    def __init__(self,*args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)

    @staticmethod
    def instance():
        return Configuration()

    def init(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        parser = argparse.ArgumentParser()
        parser.add_argument('-yaml', '--YAML_PATH', type=str, default=None)
        
        # 兼容训练作业
        parser.add_argument('-train_url', '--train_url', type=str, default=None)
        parser.add_argument('-data_url', '--data_url', type=str, default=None)
        parser.add_argument('--init_method', default=None, help='tcp_port')
        parser.add_argument('--rank', type=int, default=0, help='index of current task')
        parser.add_argument('--world_size', type=int, default=1, help='total number of tasks')
        
        args,_ = parser.parse_known_args()
        yaml_filepath = args.YAML_PATH
        print(args.__dict__)
        for k,v in args.__dict__.items():
            if v is not None:
                setattr(self,k,v)
        if yaml_filepath is not None:
            yaml_args = get_yaml_contents(yaml_filepath)
            self.update(yaml_args)
        
    def __str__(self):
        ret = "Configuration class for system config\r\n" + "=" * 10 + "\r\n"
        for key,item in self.items():
            ret += "%s:%s\r\n" % (str(key), str(item))
        ret += "=" * 10
        ret += "\r\n"
        return ret

Config = Configuration.instance()
