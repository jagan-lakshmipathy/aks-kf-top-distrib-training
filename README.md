# Distributed Training Examples Using Kubeflow Training Operator
###### Jagan Lakshmipathy 
###### 08/31/2024

### 1. Introduction
While we are focussed on distributed training of machine learning models using Kubeflow Traing Operator, we will using AKS to deploy and test this code. So, we will use Azure CLI commands with kubectl commands to control the Azure Kubernetes Service (AKS) cluster from my console. So, the steps listed here is not completely cloud provider agnostic. I am going to assume that you are going to follow the steps using AKS. However, you can follow along these steps with your preferred cloud provider for the most part with the exception of Azure CLI commands. We will show how to create a GPU nodepool to run our workload in GPUs. However, you may choose to create a CPU nodepools to run your workload as well. With this caveat let's get started.

### 2. Prerequesites
We also assume that you have a good understanding of Azure. If you would like to read about Azure please go [here](https://azure.microsoft.com/en-us/get-started). If you haven't done already installed Azure CLI, do install it as instructed in this [link](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).

We refer you to learn about Azure Kubernetes Service (AKS) from [here](https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-portal?tabs=azure-cli). Also we refer to [here](https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-portal?tabs=azure-cli) on how to request vCPU quotas from azure portal. If you would like to learn about different compute options in Azure please review this [link](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/overview?tabs=breakdownseries%2Cgeneralsizelist%2Ccomputesizelist%2Cmemorysizelist%2Cstoragesizelist%2Cgpusizelist%2Cfpgasizelist%2Chpcsizelist). In this example we will use two types of vCPUs Standard_D4ds_v5 and Standard_NC40ads_H100_v5. We will use the D4ds_v5 CPUs to run the kubernetes system workloads and NC40ads_H100_v5 CPUs to run the GPU workloads. Steps involved in requesting any other vCPUs with GPU will be very similar. In our example we run a simple Machine Learning example on the GPU.  We assume you have a reasonable understanding of Azure Cloud Platform. We are also assume you have a fairly good understanding of Kubernetes. Please do find the Kubernetes reading material from [here](https://kubernetes.io/docs/setup/). We assume that you also have a fairly good working knowledge of github. Please clone this [repo](www.github.com) to your local. Install kubectl, kubernetes cli tool, from [here](https://kubernetes.io/docs/tasks/tools/).

We will be using MacOS to run the kubernetes commands and Azure CLI commands using bash shell. You can follow along with your prefered host, operating system and shell.

### 3. What's in this Repo?
We will complete this section in the end.


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
    bash> 	az extension add --name aks-preview
	bash> 	az extension update --name aks-preview

	bash> 	az feature register --namespace "Microsoft.ContainerService" --name "GPUDedicatedVHDPreview"
	bash> 	az feature show --namespace "Microsoft.ContainerService" --name "GPUDedicatedVHDPreview"
	bash> 	az provider register --namespace Microsoft.ContainerService

```
### 6. Add nodepool to AKS Cluster

We wukk add a nodepool with 3 nodes(check Azure documentation to see the Azure's latest offering). You can choose any GPU loaded vCPU from Azure offering that you are eligible to request as per your quota requirements. I tried these GPU loaded nodes Standard_NC24s_v3, and Standard_NC40ads_H100_v5 from the NCv3-series and NCads H100 v5-series familes respectively. But the following command adds 3 40 core vCPU with 1 H100 GPU each. We can adjust the min and max counts depending on your workload. We picked a min of 1 and max of 3.

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

### 7. Create Workload Images
Let's create a docker images that we would like to run as a GPU workload. Browse the repo code and familarize yourself with the dockerfiles and the python files they refer. Note the --platform directive below that enables to generate a AMD64 based image as I was running the following command in Apple Silicon host. We will try a  

```
    bash> docker build  --platform="linux/amd64"  -t <your-tag-name:latest> .
```