import kfp
from kfp import dsl
import kfp.components as comp
from typing import NamedTuple

def train_data(with_gpu:int, print_training:int, n_iters: int, n_classes: int, batch_size: int, subdiv: int, lake_host:str, lake_user:str, lake_pwd:str, lake_repo:str, mlflow_server:str) -> str:
    import subprocess
    # install required libs
    subprocess.run(['pip', 'install', 'wget', 'gitpython', 'lakefs-client', 'mlflow==1.29.0'])

    import lakefs_client
    from lakefs_client.client import LakeFSClient
    from lakefs_client.api import objects_api
    
    # sub-functions
    def download_from_lake(host, user, pwd, repo):
        configuration = lakefs_client.Configuration()
        configuration.username = user
        configuration.password = pwd
        configuration.host = host

        client = LakeFSClient(configuration)
        # Get branch names or commit ids
        branch_and_commits = client.branches.list_branches(repo)['results']
        ref = "main" # string | branch name (id) OR a commit id (commit_id) in 'branch_and_commits' returned list

        with client.objects.api_client as api_client:
            api_instance = objects_api.ObjectsApi(api_client)

            # Get list of objects and names
            api_response = api_instance.list_objects(repo, ref)['results']
            path = "dataset.zip" # string | "path" value in returned api_response 

            # Download the object/data. It will go directly to /tmp folder
            try:
                api_response = api_instance.get_object(repo, ref, path)
            except lakefs_client.ApiException as e:
                print(str(e))

    # install darknet
    from git import Repo
    Repo.clone_from('https://github.com/pjreddie/darknet.git', 'darknet/')
    import os
    os.chdir('darknet/')
    # edit saving after per 1000 iters in detector.c
    # sed -i '138 s@i%10000==0@i%1000==0@' examples/detector.c
    subprocess.run(["sed", "-i", '138 s@i%10000==0@i%1000==0@', "examples/detector.c"])
    # edit Makefile if train with GPU and enable OPENCV
    if with_gpu:
        # sed -i 's/OPENCV=0/OPENCV=1/' Makefile
        subprocess.run(["sed", "-i", 's/OPENCV=0/OPENCV=1/', "Makefile"])
        # sed -i 's/GPU=0/GPU=1/' Makefile
        subprocess.run(["sed", "-i", 's/GPU=0/GPU=1/', "Makefile"])
        # sed -i 's/CUDNN=0/CUDNN=1/' Makefile
        subprocess.run(["sed", "-i", 's/CUDNN=0/CUDNN=1/', "Makefile"])
    subprocess.run(['make'])

    # download data
    download_from_lake(lake_host, lake_user, lake_pwd, lake_repo)

    # extract zip data to folder
    import shutil
    shutil.unpack_archive('/tmp/dataset.zip', 'data/obj/')
    
    # setup for training process
    # make a new model config file
    subprocess.run(['cp', 'cfg/yolov3-tiny.cfg', 'cfg/my_v3tiny.cfg'])
    # sed -i 's/batch=1/batch={BATCH_SIZE}/' cfg/my_v3tiny.cfg
    subprocess.run(["sed", "-i", 's/batch=1/batch={}/'.format(batch_size), "cfg/my_v3tiny.cfg"])
    # sed -i 's/subdivisions=1/subdivisions={SUB_DIVISION}/' cfg/my_v3tiny.cfg
    subprocess.run(["sed", "-i", 's/subdivisions=1/subdivisions={}/'.format(subdiv), "cfg/my_v3tiny.cfg"])
    # sed -i 's/max_batches = 500200/max_batches = {N_ITERS}/' cfg/my_v3tiny.cfg
    subprocess.run(["sed", "-i", 's/max_batches = 500200/max_batches = {}/'.format(n_iters), "cfg/my_v3tiny.cfg"])
    # sed -i 's/steps=20800,23400/steps={0.8*N_ITERS},{0.9*N_ITERS}/' cfg/my_v3tiny.cfg
    subprocess.run(["sed", "-i", 's/steps=20800,23400/steps={},{}/'.format(int(0.8*n_iters), int(0.9*n_iters)), "cfg/my_v3tiny.cfg"])
    # sed -i '135 s@classes=80@classes={N_CLASSES}@' cfg/my_v3tiny.cfg
    subprocess.run(["sed", "-i", '135 s@classes=80@classes={}@'.format(n_classes), "cfg/my_v3tiny.cfg"])
    # sed -i '177 s@classes=80@classes={N_CLASSES}@' cfg/my_v3tiny.cfg
    subprocess.run(["sed", "-i", '177 s@classes=80@classes={}@'.format(n_classes), "cfg/my_v3tiny.cfg"])
    # sed -i '127 s@filters=255@filters={(N_CLASSES+5)*3}@' cfg/my_v3tiny.cfg
    subprocess.run(["sed", "-i", '127 s@filters=255@filters={}@'.format((n_classes+5)*3), "cfg/my_v3tiny.cfg"])
    # sed -i '171 s@filters=255@filters={(N_CLASSES+5)*3}@' cfg/my_v3tiny.cfg
    subprocess.run(["sed", "-i", '171 s@filters=255@filters={}@'.format((n_classes+5)*3), "cfg/my_v3tiny.cfg"])

    # download darknet pre-trained weight
    subprocess.run(['wget', 'https://pjreddie.com/media/files/darknet53.conv.74'])

    # make .names file
    labels = ['Conn_Mod','Connector','IC','Cooler','El_Cap','XTAL_OSC','Jumper','Inductor_BIG','Transformer','Module','IC_DIL','PWR_SEMI','Conn_SubD']
    with open('pcb.names', 'w') as f:
        f.write("\n".join(labels))
    # make .data file
    with open('pcb.data', 'w') as f:
        f.write('classes = 13\ntrain = train.txt\nvalid = val.txt\nnames = pcb.names\nbackup = backup/')

    import glob
    # make a train.txt file
    train_files = "\n".join(glob.glob('data/obj/training/*.jpg'))
    with open('train.txt', 'w') as f:
        f.write(train_files)
    # make a val.txt file
    val_files = "\n".join(glob.glob('data/obj/validation/*.jpg'))
    with open('val.txt', 'w') as f:
        f.write(val_files)
    
    # train (should have ~500 - 2000 images per class + train 2000 iterations per class)
    if print_training:
        # ./darknet detector train .data_file cfg darknet53.conv.74
        subprocess.run(['./darknet', 'detector', 'train', 'pcb.data', 'cfg/my_v3tiny.cfg', 'darknet53.conv.74'])
    else:
        # ./darknet detector train data_file cfg darknet53.conv.74 &> /dev/null
        subprocess.run(['./darknet', 'detector', 'train', 'pcb.data', 'cfg/my_v3tiny.cfg', 'darknet53.conv.74', '&>' , '/dev/null'])

    # get the latest weights
    weight_list = glob.glob('backup/*.weights')
    weight_list.sort(key=lambda x:os.path.getmtime(x))
    latest_weight = weight_list[-1]

    import mlflow
    from mlflow import MlflowClient
    from mlflow.entities import ViewType

    # # set MLFlow tracking uri
    mlflow.set_tracking_uri(mlflow_server)
    # create experiment
    mlflow.create_experiment("yolo-VN", tags={"tag":"1"})
    mlflow.set_experiment("yolo-VN")
    
    mlflow.log_artifact(latest_weight)

    current_experiment=dict(mlflow.get_experiment_by_name("yolo-VN"))
    experiment_id=current_experiment['experiment_id']

    runs = MlflowClient().search_runs(experiment_ids=experiment_id, filter_string="", run_view_type=ViewType.ALL)

    return mlflow.get_artifact_uri()

train_op = comp.func_to_container_op(train_data)


def predict_data(with_gpu:int, n_classes: int, mlflow_server:str, uri:str):
    import subprocess

    # install required libs
    subprocess.run(['pip', 'install', 'gitpython', 'mlflow==1.29.0'])

    # install darknet
    from git import Repo
    Repo.clone_from('https://github.com/pjreddie/darknet.git', 'darknet/')
    import os
    os.chdir('darknet/')
    # edit Makefile if train with GPU and enable OPENCV
    if with_gpu:
        # sed -i 's/OPENCV=0/OPENCV=1/' Makefile
        subprocess.run(["sed", "-i", 's/OPENCV=0/OPENCV=1/', "Makefile"])
        # sed -i 's/GPU=0/GPU=1/' Makefile
        subprocess.run(["sed", "-i", 's/GPU=0/GPU=1/', "Makefile"])
        # sed -i 's/CUDNN=0/CUDNN=1/' Makefile
        subprocess.run(["sed", "-i", 's/CUDNN=0/CUDNN=1/', "Makefile"])
    subprocess.run(['make'])

    # setup for training process
    # make a new config file
    subprocess.run(['cp', 'cfg/yolov3-tiny.cfg', 'cfg/my_v3tiny.cfg'])

    # sed -i '135 s@classes=80@classes={N_CLASSES}@' cfg/my_v3tiny.cfg
    subprocess.run(["sed", "-i", '135 s@classes=80@classes={}@'.format(n_classes), "cfg/my_v3tiny.cfg"])
    # sed -i '177 s@classes=80@classes={N_CLASSES}@' cfg/my_v3tiny.cfg
    subprocess.run(["sed", "-i", '177 s@classes=80@classes={}@'.format(n_classes), "cfg/my_v3tiny.cfg"])
    # sed -i '127 s@filters=255@filters={(N_CLASSES+5)*3}@' cfg/my_v3tiny.cfg
    subprocess.run(["sed", "-i", '127 s@filters=255@filters={}@'.format((n_classes+5)*3), "cfg/my_v3tiny.cfg"])
    # sed -i '171 s@filters=255@filters={(N_CLASSES+5)*3}@' cfg/my_v3tiny.cfg
    subprocess.run(["sed", "-i", '171 s@filters=255@filters={}@'.format((n_classes+5)*3), "cfg/my_v3tiny.cfg"])

    import mlflow
    mlflow.set_tracking_uri(mlflow_server)

    mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path='.')

    # make .data file again
    with open('pcb.data', 'w') as f:
        f.write('classes = 13\ntrain = train.txt\nvalid = val.txt\nnames = pcb.names\nbackup = backup/')
    
    # make .names file again
    labels = ['Conn_Mod','Connector','IC','Cooler','El_Cap','XTAL_OSC','Jumper','Inductor_BIG','Transformer','Module','IC_DIL','PWR_SEMI','Conn_SubD']
    with open('pcb.names', 'w') as f:
        f.write("\n".join(labels))
    
    # predict
    # get a sample of pcb board online
    link = 'https://image.shutterstock.com/image-photo/pcb-board-integrated-circuit-600w-417842593.jpg'
    subprocess.run(['wget', '{}'.format(link)])
    # ./darknet detector test .data_file cfg weight img
    subprocess.run(['./darknet', 'detector', 'test', 'pcb.data', 'cfg/my_v3tiny.cfg', 'artifacts/my_v3tiny_final.weights', '{}'.format(link.split("/")[-1])])

pred_op = comp.func_to_container_op(predict_data)


@dsl.pipeline(
    name='Kubeflow yolo demo',
    description='Kubeflow pipeline demo with tiny yolov3'
)

def kubeflow_yolo_pipeline(
    with_gpu:int=0,
    print_training:int=1,
    n_iters:int=500, 
    n_classes:int=13, 
    batch_size:int=8, 
    subdiv:int=4,
    lake_host:str='', 
    lake_user:str='', 
    lake_pwd:str='', 
    lake_repo:str='', 
    mlflow_server:str=''):

    #Passing pipeline parameter and a constant value as operation arguments
    _train_op = train_op(with_gpu, print_training, n_iters, n_classes, batch_size, subdiv, lake_host, lake_user, lake_pwd, lake_repo, mlflow_server)
    #Returns a dsl.ContainerOp class instance.
    _pred_op = pred_op(with_gpu, n_classes, mlflow_server,_train_op.output).after(_train_op)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(kubeflow_yolo_pipeline, 'pipeline.yaml')

