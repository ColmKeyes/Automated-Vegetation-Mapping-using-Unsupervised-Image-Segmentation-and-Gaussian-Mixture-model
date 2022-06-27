
<!-- PROJECT LOGO -->
  <h3 align="center">Gaussian-Mixture-Model</h3>

<br />

<div align="center">
  <a href="https://www.wur.nl/en/research-results/chair-groups/environmental-sciences/laboratory-of-geo-information-science-and-remote-sensing.htm">
    <img src="../UISB/Logos/WUR.png" alt="WUR" width="200" height="45">
  </a>
   <a href="https://www.ecogoggle.nl/">
    <img src="../UISB/Logos/Ecogoggle.png" alt="Ecogoggle" width="200" height="60">
  </a>



<!-- ABOUT THE PROJECT -->
## About The Project
<div>
<div align="left">  
 <br />
  <br />




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
