import importlib
import copy
import hydra
import logging

from torch_points3d.datasets.base_dataset import BaseDataset

log = logging.getLogger(__name__)


def get_dataset_class(dataset_config):
    task = dataset_config.task
    
    # Find and create associated dataset
    try:
        dataset_config.dataroot = hydra.utils.to_absolute_path(
            dataset_config.dataroot)
    except Exception:
        log.error("This should happen only during testing")
    
    # class equals *_rad8.yaml's class
    dataset_class = getattr(dataset_config, "class")
    dataset_paths = dataset_class.split(".")
    module = ".".join(dataset_paths[:-1])
    class_name = dataset_paths[-1]
    # task equals *_rad8.yaml's task
    # module equls *_rad8.yaml's class's in the front of '.' ex) treeins if treeins.TreeinsFusedDataset
    if module != 'treeins':
        module = 'treeins'
    dataset_module = ".".join(["torch_points3d.datasets", task, module])
    datasetlib = importlib.import_module(dataset_module)
    print(datasetlib.__file__) # '/workspace/torch_points3d/datasets/panoptic/treeins.py'
    target_dataset_name = class_name
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset_cls = cls

    if dataset_cls is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase."
            % (module, class_name)
        )
    return dataset_cls


def instantiate_dataset(dataset_config) -> BaseDataset:
    """Import the module "data/[module].py".
    In the file, the class called {class_name}() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_cls = get_dataset_class(dataset_config)
    dataset = dataset_cls(dataset_config)
    return dataset
