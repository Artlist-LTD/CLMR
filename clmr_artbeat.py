#%%
from aistdops.config import cfg,get_secret
from aistdops.lib.clearml_datasets import AlDatasets
from aistdops.tracking import start_tracking
from clearml import Dataset
#%%
from tempfile import mkstemp, TemporaryDirectory
import os
temp_dir = TemporaryDirectory()
def set_google_creds():
    clearml_client_sa = get_secret("clearml-client-sa")
    file_tmp_path = mkstemp(
        prefix="clearml_client_sa_", suffix=".json", dir=temp_dir.name
    )[1]
    with open(file_tmp_path, "w") as f:
        f.write(clearml_client_sa)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = file_tmp_path

set_google_creds()
# %%

project_name="CLMR"
experiment_name="sample on CLMR"
cmlogger, task = start_tracking(
    experiment_name=experiment_name, 
    project_name=project_name,
)

# %%
ds = Dataset.get(dataset_name = "samples", dataset_project="artbeat-data")
dataset_path=ds.get_local_copy()

from preprocess import main as preprocess_main
from argparse import ArgumentParser
from main import main as main_m
from pytorch_lightning import Trainer
from clmr.utils import yaml_config_hook

def p_main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="audio")
    parser.add_argument("--dataset_dir", type=str, default=dataset_path)
    parser.add_argument("--sample_rate", type=int, default=22050)
    args = parser.parse_args()

    preprocess_main(args)
    return "done"

def m_main():
    parser = ArgumentParser(description="CLMR")
    parser = Trainer.add_argparse_args(parser)
    config = yaml_config_hook("./config/config.yaml")
    config["dataset"]="audio"
    config["dataset_dir"]=dataset_path
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    main_m(args)
    return "main done"

if __name__ == "__main__":
    res=p_main()
    res_main=m_main()
    print(res_main)
