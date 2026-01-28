# vhr-cloudmask.pytorch

[![DOI](https://zenodo.org/badge/942722065.svg)](https://doi.org/10.5281/zenodo.18406517)

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

## Quickstart

To run inference for all "tif" files in a directory:

```bash
singularity shell --nv --env PYTHONPATH=/explore/nobackup/people/$USER/development/VHR-TOOLKIT-FRAMEWORK/vhr-cloudmask.pytorch -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects,/css,/nfs4m /lscratch/$NOBACKUP/container/vhr-cloudmask.pytorch python /explore/nobackup/people/$NOBACKUP/development/VHR-TOOLKIT-FRAMEWORK/vhr-cloudmask.pytorch/vhr_cloudmask/view/cloudmask_cnn_pipeline_cli.py -r '/explore/nobackup/projects/above/misc/ABoVE_Shrubs/srlite/002m/5-toas/*.tif' -o 'vhr-cloudmask-outputs' --overwrite
```

## Authors

- Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov

## Contributing

Please see our [guide for contributing to vhr-cloudmask.pytorch](CONTRIBUTING.md). Contributions
are welcome, and they are greatly appreciated! Every little bit helps, and credit will
always be given.

You can contribute in many ways:

### Report Bugs

Report bugs at https://github.com/nasa-nccs-hpda/vhr-cloudmask.pytorch/issues.

If you are reporting a bug, please include:
- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and
"help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is
open to whoever wants to implement it.

### Write Documentation

vhr-cloudmask.pytorch could always use more documentation, whether as part of the official vhr-cloudmask.pytorch docs,
in docstrings, or even on the web in blog posts, articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/nasa-nccs-hpda/vhr-cloudmask.pytorch/issues.

If you are proposing a feature:
- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions are welcome :)

## Additional References

[1] Raschka, S., Patterson, J., & Nolet, C. (2020). Machine learning in python: Main developments and technology trends in data science, machine learning, and artificial intelligence. Information, 11(4), 193.

[2] Paszke, Adam; Gross, Sam; Chintala, Soumith; Chanan, Gregory; et all, PyTorch, (2016), GitHub repository, <https://github.com/pytorch/pytorch>. Accessed 13 February 2020.

[3] Caraballo-Vega, J., Carroll, M., Li, J., & Duffy, D. (2021, December). Towards Scalable & GPU Accelerated Earth Science Imagery Processing: An AI/ML Case Study. In AGU Fall Meeting 2021. AGU.

[4] Jordan Alexis Caraballo-Vega. (2026). nasa-nccs-hpda/vhr-cloudmask.pytorch: 0.1.0 (0.1.0). Zenodo. https://doi.org/10.5281/zenodo.18406518