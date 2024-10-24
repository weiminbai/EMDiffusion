## Codebase for reviewers

### Environment

Since the E-step is the same as [DPS](https://github.com/DPS2022/diffusion-posterior-sampling), so our required environment is the same as DPS, which is simple and easy to set.

The difference is that we adopt the Accelerate package for distributed training, you could easily install one that matches your machine. Or you can simply delete it in the code, then train diffusion with one GPU.

We perform E-steps and M-steps iteratively, specifically:

**Perform E-step:**

`bash e-step.sh`

**Perform M-step:**

`bash m-step.sh`

Please remember to change some hyperparameters defined in the two shell script that relate to the model path, dataset path or saving path, etc.