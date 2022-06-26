<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->



<!-- PROJECT LOGO -->

  <h3 align="center">Unsupervised Image Segmentation by Backpropegation</h3>

<br />

<div align="center">
  <a href="https://www.wur.nl/en/research-results/chair-groups/environmental-sciences/laboratory-of-geo-information-science-and-remote-sensing.htm">
    <img src="Logos/WUR.png" alt="WUR" width="200" height="45">
  </a>
   <a href="https://www.ecogoggle.nl/">
    <img src="Logos/Ecogoggle.png" alt="Ecogoggle" width="200" height="60">
  </a>


  ![](Tiger_Segmentation_Example.gif)



<!-- ABOUT THE PROJECT -->
## About The Project
<div>
<div align="left">  
CNN for unsupervised image segmentation. For unsupervised methods, a series of learning criteria must be set. This model optimises predictions through backpropegation and is based on the three following criteria: <br />
  <br />

  (a) Pixels of similar features are desired to be assigned the same label.<br />
  (b) Spatially continuous pixels are desired to be assigned the same label.<br />
  (c) The number of unique cluster labels is desired to be large.<br />

### Prerequisites


    
* [pytorch](https://pytorch.org/)
  

  
* [CUDA](https://developer.nvidia.com/cuda-toolkit)

<div>
  
### Installation


  
  1.Get started with Pytorch <br />
`https://pytorch.org/get-started/locally/` <br />
`https://varhowto.com/install-pytorch-cuda-10-0/`  <br />
  
  2.Clone the repo <br />
`git clone https://github.com/ColmKeyes/Automated-Vegetation-Mapping-using-Unsupervised-Image-Segmentation-and-Gaussian-Mixture-model.git` <br />
 



<!-- USAGE EXAMPLES -->
## Usage

This script can be initiated from Terminal using

`python UISB.py --minLabels 10 --compactness 0.1 --lr 0.1 --nConv 2 --input images/108004.jpg`

Or by running the script directly in python using and changing the parameters under the Inits section.

<!-- Future Work -->
## Future Work

- [ ] Consistent Seed
- [ ] consistent Colourmap usage
- [ ] move preprocessing to within script
- [ ] Loss function
    - [ ] Investigate other torch loss functions
    - [ ] Create custom loss function for given problem
- [ ] Validation
    - [ ] Implement GT validation in script
    - [ ] Automate Validation
- [ ] Metrics
    - [ ] Add Entropy evaluation to script
    - [ ] Add chi evaluation to script
  
<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.



<!-- CONTACT -->
## Contact

[Colm Keyes](https://www.linkedin.com/in/colm-keyes-4960a5132/) - keyesco@tcd.ie

  
<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
Acknowledgments go out to the original creator of this network, [Asako Kanezak](https://github.com/kanezaki), AIST, Tokyo,
who's informative paper can be found here: https://ieeexplore.ieee.org/document/8462533

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
