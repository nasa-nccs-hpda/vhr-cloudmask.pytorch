# vhr-cloudmask.pytorch

Expanding our current cloud masking capabilities using OmniCloudMask

## Downloading the Container

All Python and GPU depenencies are installed in an OCI compliant Docker image. You can
download this image into a Singularity format to use in HPC systems.

```bash
singularity pull docker://nasanccs/vhr-cloudmask.pytorch:latest
```

In some cases, HPC systems require Singularity containers to be built as sandbox environments because
of uid issues (this is the case of NCCS Explore). For that case you can build a sandbox using the following
command. Depending the filesystem, this can take between 5 minutes to an hour.

```bash
singularity build --sandbox /lscratch/jacaraba/container/vhr-cloudmask.pytorch docker://nasanccs/vhr-cloudmask.pytorch:latest
```

If you have done this step, you can skip the Installation step since the containers already
come with all dependencies installed.

## Initial Testing

Generated a test script for now that uses their implementation. Submitted a PR to fix the get_model
function from having a static path. Once that is implemented, we will include the pipeline as 
an operational pipeline to feed into the vhr-toolkit. For now, to run the current example:

```bash
PYTHONPATH="/explore/nobackup/people/jacaraba/development/vhr-cloudmask.pytorch/OmniCloudMask" python test_omnicloudmask.py
```

singularity shell --nv --env PYTHONPATH=/explore/nobackup/people/$USER/development/VHR-TOOLKIT-FRAMEWORK/vhr-cloudmask.pytorch -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects,/css,/nfs4m /lscratch/jacaraba/container/vhr-cloudmask.pytorch