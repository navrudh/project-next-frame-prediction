# import getpass
# import json
# import os
#
# from torchvision.datasets.folder import make_dataset
# from torchvision.datasets.utils import list_dir

# username = getpass.getuser()
# config = json.load(open(f"{username}.json"))
# CLASSIFICATION_DATASET = config["classification"]["root"]
# CHECKPOINT_PATH = config["prediction"]["model"]
# extensions = ('avi',)
# root_path = config["ucf101"]["root"]
# anno_path = config["ucf101"]["anno"]
# classes = list(sorted(list_dir(root=root_path)))
# class_to_idx = {classes[i]: i for i in range(len(classes))}
# train_samples = make_dataset(root_path, class_to_idx,
#                              extensions, is_valid_file=None)
# video_list = [x[0] for x in train_samples]
#
#
# def _select_fold( video_list, annotation_path, fold, train):
#     name = "train" if train else "test"
#     name = "{}list{:02d}.txt".format(name, fold)
#     f = os.path.join(annotation_path, name)
#     selected_files = []
#     with open(f, "r") as fid:
#         data = fid.readlines()
#         data = [x.strip().split(" ") for x in data]
#         data = [os.path.join(self.root, x[0]) for x in data]
#         selected_files.extend(data)
#     selected_files = set(selected_files)
#     indices = [i for i in range(len(video_list)) if video_list[i] in selected_files]
#     return indices
from train_video_prediction import SelfSupervisedVideoPredictionLitModel

lit_model = SelfSupervisedVideoPredictionLitModel(
    hidden_dims=[64, 64, 128, 256], batch_size=8
)

lit_model.train_dataloader()
lit_model.test_dataloader()
