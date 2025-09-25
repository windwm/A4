import json
import copy
import mindspore.common.dtype as mstype

__all__=['AbsModelConfig', 'ModelConfig']

class AbsModelConfig(object):
    def __init__(self):
        pass

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `ModelConfig` from a Python dictionary of parameters."""
        config = cls()
        for key, value in json_object.items():
            if isinstance(value, dict):
                value = AbsModelConfig.from_dict(value)
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `ModelConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        def _json_default(obj):
            if isinstance(obj, AbsModelConfig):
                return obj.__dict__
        return json.dumps(self.__dict__, indent=2, sort_keys=True, default=_json_default) + "\n"

class ModelConfig(AbsModelConfig):
    """Configuration class to store the configuration of a :class:`~DeBERTa.deberta.DeBERTa` model.
        Attributes:
        
    """
    def __init__(self):
        """Constructs ModelConfig.
        Set Model Hyper-parameters:
        """
        self.dropout_rate = 0.1
        self.act_func = "relu" ### "relu" or "gelu"
        self.num_intermediate_factor = 4

        # ### Debug @ PYNATIVE_MODE:
        # self.msint = mstype.int32
        # self.msfp = mstype.float32 ### Debug.
        # self.ms_small = 1e-8 ### In case of log(0) or divide by 0;
        # self.recompute = False ### Debug @ PYNATIVE_MODE
        # self.distributed = False ### True when train

        # ### Debug @ PYNATIVE_MODE:
        # self.msint = mstype.int32
        # self.msfp = mstype.float16 ### Debug.
        # self.ms_small = 1e-5 ### In case of log(0) or divide by 0;
        # self.recompute = False ### Debug @ PYNATIVE_MODE
        # self.distributed = False ### True when train

        #         ### Debug @ GRAPH_MODE:
        #         self.msint = mstype.int32
        #         self.msfp = mstype.float16
        #         self.ms_small = 1e-5 ### In case of log(0) or divide by 0;
        #         self.recompute = True ### Train @ GRAPH_MODE
        #         self.distributed = False ### True when train

        ### Distributed Training:
        self.msint = mstype.int32
        self.msfp = mstype.float16
        self.ms_small = 1e-5 ### In case of log(0) or divide by 0;
        # self.recompute = True ### Train @ GRAPH_MODE
        self.recompute = False
        self.distributed = True ### Train or Inference at Clusters
