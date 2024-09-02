# Distributed Training Examples Using Kubeflow Training Operator
###### Jagan Lakshmipathy 
###### 08/31/2024

### 1. Introduction
In one of our earlier [work](https://github.com/jagan-lakshmipathy/aks-gpu-job), we demonstrated a step-by-step process on how to run a simple machine learning workload in a GPU enhanced vCPU in Azure Kubernetes (AKS). Here we will focus on how to distribute training of machine learning models using  [Kubeflow Traing Operator](https://www.kubeflow.org/docs/components/training/).We will use AKS to deploy and test this code. So, we will use Azure CLI commands with kubectl commands to control the Azure Kubernetes Service (AKS) cluster from our console. So, the steps listed here is not completely cloud provider agnostic. We are going to assume that you are going to follow along using AKS. However, you can follow along with any of your preferred cloud provider for the most part with the exception of Azure CLI commands. We will show how to create a GPU nodepool to run our workload in GPUs. However, you may choose to create a CPU nodepools to run your workload as well. Let's get started.

### 2. Prerequesites
We also assume that you have a good understanding of Azure. If you would like to read about Azure please go [here](https://azure.microsoft.com/en-us/get-started). If you haven't done already installed Azure CLI, do install it as instructed in this [link](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).

We refer you to learn about Azure Kubernetes Service (AKS) from [here](https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-portal?tabs=azure-cli). Also we refer to [here](https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-portal?tabs=azure-cli) on how to request vCPU quotas from azure portal. If you would like to learn about different compute options in Azure please review this [link](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/overview?tabs=breakdownseries%2Cgeneralsizelist%2Ccomputesizelist%2Cmemorysizelist%2Cstoragesizelist%2Cgpusizelist%2Cfpgasizelist%2Chpcsizelist). In this example we will use two types of vCPUs Standard\_D4ds\_v5 and Standard\_NC40ads\_H100\_v5. We will use the D4ds\_v5 CPUs to run the kubernetes system workloads and NC40ads\_H100\_v5 CPUs to run the GPU workloads. Steps involved in requesting any other vCPUs with GPU will be very similar. In our example we run a simple Machine Learning example on the GPU.  We assume you have a reasonable understanding of Azure Cloud Platform. We are also assume you have a fairly good understanding of Kubernetes. Please do find the Kubernetes reading material from [here](https://kubernetes.io/docs/setup/). We assume that you also have a fairly good working knowledge of github. Please clone this [repo](www.github.com) to your local. Install kubectl, kubernetes cli tool, from [here](https://kubernetes.io/docs/tasks/tools/).

We will be using MacOS to run the kubernetes commands and Azure CLI commands using bash shell. You can follow along with your prefered host, operating system and shell.

### 3. What's in this Repo?
This repo has a docker file to build the workload image, training workload in mnist\_entropy.py, and a manifest to deploy this workload.


### 4. Authenticate Your Console
We assume that the kubernetes cluster is up and running. We will do the following two steps to prepare our console to be authenticated to interact with the AKS cluster remotely.

1. Login to your Azure Portal and make sure the kubernetes cluster is up and running.You can also check the cluster from your bash console. For that to work we need to have the *kubectl* working. So go to Step 2, before you try out any *kubectl* commands.

2. In order to issue kubectl commands to control AKS cluster from your local console we need to merge the credentials with the local kube config. Kubernetes config file is typically located under /Users/\<username\>/.kube/config in MacOS. The following azure cli command would merge the config. The second command lets you see the running pods in the cluster:

```
    bash> az aks get-credentials --resource-group <resource-group-name> --name <aks-cluster-name>
    bash> kubectl get pods --watch

```

### 5. Register Microsoft Container Service
We will issue the following Azure CLI commands to register the container service.
```
    bash>   az extension add --name aks-preview
    bash>   az extension update --name aks-preview

    bash>   az feature register --namespace "Microsoft.ContainerService" --name "GPUDedicatedVHDPreview"
    bash>   az feature show --namespace "Microsoft.ContainerService" --name "GPUDedicatedVHDPreview"
    bash>   az provider register --namespace Microsoft.ContainerService

```
### 6. Add nodepool to AKS Cluster

We will add a nodepool with 3 nodes(check Azure documentation to see the Azure's latest offering). You can choose any GPU loaded vCPU from Azure offering that you are eligible to request as per your quota requirements. I tried these GPU loaded nodes Standard\_NC24s\_v3, and Standard\_NC40ads\_H100\_v5 from the NCv3-series and NCads H100 v5-series familes respectively. But the following command adds 3 40 core vCPU with 1 H100 GPU each. We can adjust the min and max counts depending on your workload. We picked a min of 1 and max of 3. This command also taints the nodes with key and value with 'sku' and 'gpu' respectively.

```
    bash> az aks nodepool add --resource-group <name-of-resource-group> --cluster-name <cluster-name> --name <nodepool-name> --node-count 2 --node-vm-size Standard_NC40ads_H100_v5 --node-taints sku=gpu:NoSchedule --aks-custom-headers UseGPUDedicatedVHD=true --enable-cluster-autoscaler --min-count 1 --max-count 3

```

### 7. Create a Azure Container Registry
We need an image to run as a workload in AKS. Also, We would need a Azure Container Registry (ACR) to push your image. So, lets create a ACR, if you don't have it already. Here is the command:

```
    bash> az acr create --name <name-of-acr> --resource-group <resource-group-associated> --sku basic
```

### 8. Login to ACR
Now lets login to the ACR before you can upload any images to ACR.

```
    bash> az acr login --name <name-of-acr>
```

### 7. Create Workload Image Locally
Let's create a docker images that we would like to run as a GPU workload. This repo contains a dockerfile named Dockerfile.ce. At line # 1, this docker file pulls a PyTorch container base image from NVIDIA. Tag 24.07-py3 is the latest available at the time of this writing. This container image contains the complete source of the version of PyTorch in /opt/pytorch. It is a prebuild and installed in the default environment (/usr/local/lib/python3.10/dist-packages/torch). This container also includes the following pacakges: (a) Pyton 3.10, (b) CUDA, (c) NCCL backend, (d) JupyterLab and beyond. Please look at this link for more details [here](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-07.html). This docker file also copies mnist\_entropy.py from current directory to working directory. The mnist_entropy.py is the python training workload we will distribute to demonstrate the distribution. In the end we have a entrypoint which we overwrite when we execute it through training operator. So, we can safely ingore. The following command will create a pytorch-mnist-ce-distributed image to the local docker registry.

```
    bash> docker build  --platform="linux/amd64"  -t pytorch-mnist-ce-distributed:1.0 .
```
### 8. Python Training Workload
Feel free to browse mnist\_entropy.py provided in this repo. This is a simple CNN we train to classify the CIFAR10 dataset. While this is a standard CNN code, the following are some points that are pertinent to training distribution. This code will be deployed in AKS using the PyTorchJob manifest pytorch\_job\_nccl\_entropy.yaml. We will go over this manifest in detail (see section 11 below). PyTorchJob is a [CRD](https://docs.openshift.com/container-platform/3.11/admin_guide/custom_resource_definitions.html) defined as a part of Kubeflow Training Operator. This CRD leverages the torchrun under the hood. The torchrun tool sets the required envornment variables (e.g. RANK, WORLD_SIZE, etc.) and runs the workload as specified in the manifests (see section below for detail). Our manifests, basically creates a master and two workers and these workloads will run on nodes that are tainted with key and value as "sku" and "gpu" respectively.

### 9. Tag and push the image to ACR
Now that we have created an image in the local registry, we need to push this image to the ACR before it can be run in the AKS. First, we need to tag the image and then we push the tagged image to an already created ACR. The following are the commands in that order.
```
    bash> docker tag pytorch-mnist-ce-distributed:1.0 <acr-name>.azurecr.io/pytorch-mnist-ce-distributed:latest
    bash> docker push <acr-name>.azurecr.io/pytorch-mnist-ce-distributed:latest
```

### 10. Attach the ACR to the AKS Cluster
Now that we have pushed the image to the ACR, we have to now attach that ACR to the cluster so that our job can access the ACR to pull the image from. We use the following command.

```
    bash> az aks update --name <aks-cluster-name> --resource-group <aks-rg-name>  --attach-acr <name-of-acr-to-attach>
```
### 11. PyTorchJob Manifest
In this repo, we have provided a manifest named pytorch\_job\_nccl\_entropy.yaml. This is a CRD provided with the training operator. So, we need to install the training operator prior to deploying this manifest in AKS. This CRD is called PyTorchJob. Under the hood training operator creates a master and workers as specfied by the PyTorchJob.pytorchReplicaSpecs. Runtime arguments for running container image and arguments as specfied by PytorchJob.spec.pytorchReplicaSpecs.spec.master.cotainers and PytorchJob.spec.pytorchReplicaSpecs.spec.worker.cotainers. The master.spec.replicas and worker.spec.replicas determine the number instances of master and workers respectively. The tolerations section to filter the tainted nodes for deployment. 

### 10. Install Kubeflow Training Operator
We will install the training operator in AKS using the following command. Please note we are deploying the version 1.8.0. This is the latest version as of this writing.
```
kubectl create -k "github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.8.0"
```
### 11. Deploy PyTorchJob
Deploy the PyTorchJob as follows. This manifest will run pytorch-mnist-ce-distributed image in 3 pods (1 master and 2 workers) by default. Feel free to edit this file as needed. Also, make sure to edit the PyTorchJob.spec.pytorchReplicaSpecs.Master.spec.image and PyTorchJob.spec.pytorchReplicaSpecs.Worker.spec.image fields accordingly. This make take a few minutes depending on how long it takes to initialize the pods. Sometimes, worker pods may timeout and post some errors. Don't worry, the pods will self heal once the master becomes available. After that all the pods will work together and will complete the task. We have shown the output of the master and two workers below.

```
    kubectl apply -f pytorch_job_nccl_entropy.yaml



----Master 0 Output -----------------
train_loss 2.069335988451781  epoch:  0  rank:  0  local_rank:  0
val_loss 1.7140780908107758  epoch:  0  rank:  0  local_rank:  0
val_acc 0.372  epoch:  0  rank:  0  local_rank:  0
1667it [00:10, 153.99it/s]
train_loss 1.555097908371569  epoch:  1  rank:  0  local_rank:  0
val_loss 1.4838463139057159  epoch:  1  rank:  0  local_rank:  0
val_acc 0.4617  epoch:  1  rank:  0  local_rank:  0
1667it [00:10, 154.90it/s]
train_loss 1.407126972989258  epoch:  2  rank:  0  local_rank:  0
val_loss 1.3695419405460358  epoch:  2  rank:  0  local_rank:  0
val_acc 0.5125  epoch:  2  rank:  0  local_rank:  0
1667it [00:10, 153.71it/s]
train_loss 1.2971398550625968  epoch:  3  rank:  0  local_rank:  0
val_loss 1.2946766710758209  epoch:  3  rank:  0  local_rank:  0
val_acc 0.5432  epoch:  3  rank:  0  local_rank:  0
1667it [00:10, 156.65it/s]
train_loss 1.2047319369235294  epoch:  4  rank:  0  local_rank:  0
val_loss 1.2222573731899262  epoch:  4  rank:  0  local_rank:  0
val_acc 0.5685  epoch:  4  rank:  0  local_rank:  0
1667it [00:10, 158.57it/s]
train_loss 1.1312249645760621  epoch:  5  rank:  0  local_rank:  0
val_loss 1.1862873133182525  epoch:  5  rank:  0  local_rank:  0
val_acc 0.5827  epoch:  5  rank:  0  local_rank:  0
1667it [00:10, 153.26it/s]
train_loss 1.0692664025944725  epoch:  6  rank:  0  local_rank:  0
val_loss 1.1657427067279815  epoch:  6  rank:  0  local_rank:  0
val_acc 0.5917  epoch:  6  rank:  0  local_rank:  0
1667it [00:10, 161.34it/s]
train_loss 1.011953621959715  epoch:  7  rank:  0  local_rank:  0
val_loss 1.1706829908251761  epoch:  7  rank:  0  local_rank:  0
val_acc 0.5944  epoch:  7  rank:  0  local_rank:  0
1667it [00:10, 154.83it/s]
train_loss 0.9644719697557386  epoch:  8  rank:  0  local_rank:  0
val_loss 1.2004791494369507  epoch:  8  rank:  0  local_rank:  0
val_acc 0.5905  epoch:  8  rank:  0  local_rank:  0
1667it [00:10, 159.13it/s]
train_loss 0.9194450580884256  epoch:  9  rank:  0  local_rank:  0
val_loss 1.2093328654766082  epoch:  9  rank:  0  local_rank:  0
val_acc 0.593  epoch:  9  rank:  0  local_rank:  0
test_acc 0.6064  epoch:  9  rank:  0  local_rank:  0

----Worker 0 Output -----------------
train_loss 2.069935476558253  epoch:  0  rank:  1  local_rank:  0
val_loss 1.7140780908107758  epoch:  0  rank:  1  local_rank:  0
val_acc 0.372  epoch:  0  rank:  1  local_rank:  0
1667it [00:10, 155.36it/s]
train_loss 1.5461383531318142  epoch:  1  rank:  1  local_rank:  0
val_loss 1.4838463139057159  epoch:  1  rank:  1  local_rank:  0
val_acc 0.4617  epoch:  1  rank:  1  local_rank:  0
1667it [00:10, 154.36it/s]
train_loss 1.400046923343622  epoch:  2  rank:  1  local_rank:  0
val_loss 1.3695419405460358  epoch:  2  rank:  1  local_rank:  0
val_acc 0.5125  epoch:  2  rank:  1  local_rank:  0
1667it [00:10, 155.93it/s]
train_loss 1.2974522332803318  epoch:  3  rank:  1  local_rank:  0
val_loss 1.2946766710758209  epoch:  3  rank:  1  local_rank:  0
val_acc 0.5432  epoch:  3  rank:  1  local_rank:  0
1667it [00:10, 156.48it/s]
train_loss 1.2095763878008527  epoch:  4  rank:  1  local_rank:  0
val_loss 1.2222573731899262  epoch:  4  rank:  1  local_rank:  0
val_acc 0.5685  epoch:  4  rank:  1  local_rank:  0
1667it [00:10, 158.08it/s]
train_loss 1.1378660896311423  epoch:  5  rank:  1  local_rank:  0
val_loss 1.1862873133182525  epoch:  5  rank:  1  local_rank:  0
val_acc 0.5827  epoch:  5  rank:  1  local_rank:  0
1667it [00:10, 154.70it/s]
train_loss 1.0765880917226331  epoch:  6  rank:  1  local_rank:  0
val_loss 1.1657427067279815  epoch:  6  rank:  1  local_rank:  0
val_acc 0.5917  epoch:  6  rank:  1  local_rank:  0
1667it [00:10, 161.57it/s]
train_loss 1.0205440365166885  epoch:  7  rank:  1  local_rank:  0
val_loss 1.1706829908251761  epoch:  7  rank:  1  local_rank:  0
val_acc 0.5944  epoch:  7  rank:  1  local_rank:  0
1667it [00:10, 154.27it/s]
train_loss 0.9742585511606614  epoch:  8  rank:  1  local_rank:  0
val_loss 1.2004791494369507  epoch:  8  rank:  1  local_rank:  0
val_acc 0.5905  epoch:  8  rank:  1  local_rank:  0
1667it [00:10, 158.38it/s]
train_loss 0.9300361643151459  epoch:  9  rank:  1  local_rank:  0
val_loss 1.2093328654766082  epoch:  9  rank:  1  local_rank:  0
val_acc 0.593  epoch:  9  rank:  1  local_rank:  0
test_acc 0.6064  epoch:  9  rank:  1  local_rank:  0

------ Worker 1 Output -----------------------------
train_loss 2.0716924758892827  epoch:  0  rank:  2  local_rank:  0
val_loss 1.7140780908107758  epoch:  0  rank:  2  local_rank:  0
val_acc 0.372  epoch:  0  rank:  2  local_rank:  0
1667it [00:10, 155.12it/s]
train_loss 1.562211665659517  epoch:  1  rank:  2  local_rank:  0
val_loss 1.4838463139057159  epoch:  1  rank:  2  local_rank:  0
val_acc 0.4617  epoch:  1  rank:  2  local_rank:  0
1667it [00:10, 155.06it/s]
train_loss 1.413986275992711  epoch:  2  rank:  2  local_rank:  0
val_loss 1.3695419405460358  epoch:  2  rank:  2  local_rank:  0
val_acc 0.5125  epoch:  2  rank:  2  local_rank:  0
1667it [00:10, 154.08it/s]
train_loss 1.304555330686964  epoch:  3  rank:  2  local_rank:  0
val_loss 1.2946766710758209  epoch:  3  rank:  2  local_rank:  0
val_acc 0.5432  epoch:  3  rank:  2  local_rank:  0
1667it [00:10, 155.93it/s]
train_loss 1.2175320370975338  epoch:  4  rank:  2  local_rank:  0
val_loss 1.2222573731899262  epoch:  4  rank:  2  local_rank:  0
val_acc 0.5685  epoch:  4  rank:  2  local_rank:  0
1667it [00:10, 156.98it/s]
train_loss 1.1497726011540836  epoch:  5  rank:  2  local_rank:  0
val_loss 1.1862873133182525  epoch:  5  rank:  2  local_rank:  0
val_acc 0.5827  epoch:  5  rank:  2  local_rank:  0
1667it [00:10, 155.29it/s]
train_loss 1.0878734888523443  epoch:  6  rank:  2  local_rank:  0
val_loss 1.1657427067279815  epoch:  6  rank:  2  local_rank:  0
val_acc 0.5917  epoch:  6  rank:  2  local_rank:  0
1667it [00:10, 159.15it/s]
train_loss 1.030855577427896  epoch:  7  rank:  2  local_rank:  0
val_loss 1.1706829908251761  epoch:  7  rank:  2  local_rank:  0
val_acc 0.5944  epoch:  7  rank:  2  local_rank:  0
1667it [00:10, 154.11it/s]
train_loss 0.9802147078170845  epoch:  8  rank:  2  local_rank:  0
val_loss 1.2004791494369507  epoch:  8  rank:  2  local_rank:  0
val_acc 0.5905  epoch:  8  rank:  2  local_rank:  0
1667it [00:10, 157.96it/s]
train_loss 0.9316146925994573  epoch:  9  rank:  2  local_rank:  0
val_loss 1.2093328654766082  epoch:  9  rank:  2  local_rank:  0
val_acc 0.593  epoch:  9  rank:  2  local_rank:  0
test_acc 0.6064  epoch:  9  rank:  2  local_rank:  0
```
### 12. Job Monitoring Commands:
```
    kubectl get pods --watch
    kubectl logs <pod-id>
    kkubectl describe pod <pod-id>
```
### 13. Tear-down
Once you are successfully done testing this code, make sure to cleanup the jobs in AKS. Finally, don't forget to tear-down the AKS cluster to avoid incurring unnecessary billing costs.
