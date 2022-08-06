## Latent Variables in Computer Vision

The goal of deep generative models is to take as input a training sample from some distribution and learn a model that represents that distribution.
A good generative model should have the following two capabilities:
1. It should be able to generate a random new image which not present in the training dataset, but looks similar to the training data images.
2. They should be able to alter or explore variations on the data we already have in specific direction.

To satisfy the second capability, a generative model should have a continous latent space which will allow smooth interpolation.

Next we will take a look at some of the generative models used in field of computer vision.

### AutoEncoder (AE)

AutoEncoders uses an unsupervised approach for learning a lower-dimensional feature representation from unlabeled training data. The encoder block learns the mapping from the input data to a low-dimensional latent space z. And the decoder recreates the orginal image from the latent vector. Autoencoders are optimized on re-construction loss. 

Limitation:
- Latent space is not continous. Because of this smooth interpolation is not possible and hence the capability to generate new images are limited.
- There are empty spaves in the latent spaces. If a point from this empty space is given as input to the decoder it will generate unrealistic images.

### Variational AutoEncoder (VAE)

VAE is designed to overcome some of the above limiatation of the AutoEncoder. Latent space of VAE are, by design, continous, allowing easy random sampling and interpolation.
- Write how it is able to achieve this
- Figure to compare the latent spave of both AE and VAE

### General Adveserial Networks (GAN)

Key idea
Figure
Generation process

#### My experiments with the latent space of GAN 

##### Training a GAN network

- Dataset
- Generator architecture
- Discriminator architecture
- Training details

Link to the Kaggle Notebook

- Generating new images
- Linear interpolation
- Controlling the generation

#### Other tools for exploring latent space

##### Latent Compass

##### Steerable GAN


#### Further Readings



You can use the [editor on GitHub](https://github.com/raigon44/xai.github.io/edit/main/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/raigon44/xai.github.io/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
