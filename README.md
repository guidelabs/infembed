# <span style="color:black"> ♾️ Embed </span>

![infembed Logo](./docs/static/img/banner.png)

InfEmbed is an error detection library for machine learning models (classifiers and generative models) built in PyTorch. Given a test dataset, InfEmbed allows you to find the groups of samples in the test data on which your model makes mistakes (errors).  

**InfEmbed is currently under actively development, so we welcome bug reports, pull requests, and your feedback.**

## Installation
We recommend that you setup a conda or virtual environment for your analysis.

### Installation Requirements
- Python >= 3.9
- PyTorch >= 1.12

**Manual install via cloning the repo**
```sh
git clone https://github.com/guidelabs/infembed.git

# now change into the infembed directory
cd infembed

# install the package
pip install .
```

## Getting Started and Tutorial
In this section, we will go over a simple tutorial setting to demonstrate the capabilities of the InfEmbed package.

## FAQ
- How is error finding different from applying an explanability method? 
    - An explanability method typically identifies the most important features (or data samples) that a model is sensitive to for its output. This is different from whether the model is more likely to make a mistake on that sample.
- Do you support generative models?
    - Yes, our approach translates in a straightforward manner to generative models on images, and large language models trained on text. We will provide examples on how to apply the package for both settings in upcoming updates.
- What does is mean for a generative model to make a mistake (or error)? 
    - This typically means that the model has low performance when it generates outputs with similar features to the sample under consideration. For example, it might be the case that an image generative model generates blurry outputs for Fruits. It could be that a large language model is more likely to hallucinate on a particular type of prompt.

## References
Infembed implements several embedding algorithms. 

* Our implementation approach is inspired by the [Captum](https://github.com/pytorch/captum). However, we focus on error discovery.

* `ArnoldiEmbedder` is based on the work of [Wang et. al. (2023)](https://arxiv.org/abs/2312.04712). It uses a Arnoldi iteration to compute an approximation to the Hessian.

* `FastKFACEmbedder` is based on the work of [Gross, Bae, & Anil et. al. (2023)](https://arxiv.org/abs/2308.03296). It uses a KFAC approximation to the Hessian, which makes the following approximations: it computes the Gauss-Newton Hessian (GNH), which is always guaranteed to be positive semi-definitive. The GNH is assumed to be block-diagonal, where the blocks correspond to parameters from different layers.

* `GradientEmbedder` is based on the work of [Zeng et. al. (2023)](https://arxiv.org/abs/2210.06759). It use a PCA projected version of the gradients for determining the error clusters.

## License
InfEmbed is distributed under the BSD license.