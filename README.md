[![Python 3.8.8+](https://img.shields.io/badge/python-3.8.8+-blue.svg)](https://www.python.org/downloads/)
[![BSD-4-Clause](https://img.shields.io/github/license/ChristianGerloff/interbrain-network-embedding)](https://img.shields.io/github/license/ChristianGerloff/interbrain-network-embedding)
[![DOI: 10.1002/hbm.25966](https://img.shields.io/badge/DOI-10.1002%2Fhbm.25966-blue)](https://doi.org/10.1002/hbm.25966)

# Demo of "Interacting brains revisited: A cross-brain network neuroscience perspective"

This repository is a demo for the interbrain networks and embedding method described in "Interacting brains revisited: A cross-brain network neuroscience perspective".

---


## <a id='overview'></a> Overview:

  The paper posits a comprehensive analytical framework for inference and prediction based on bipartite interbrain graphs and shows the general topological properties of interbrain networks. The proposed embedding in the paper provides a method to study interpersonal synchrony for inferential inquiries and prediction on individual level. This repository illustrates the generation of interbrain networks and the interbrain network embedding. The proposed framework comes with the following benefits:

*   interpretable: yields both global and region-specific insight (which regions seed  interpersonal neural synchrony)
*   accounts for uncertainty in channel localization
*   is applicable for both directed and undirected connectivity estimators

## Notebook

The repository contains an interactive notebook describing graph construction, graph reduction, and the embedding for ease of use.

---

## <a id='dependencies'></a> Dependencies:

The repo depends on Python and requires Python >= 3.8.8. <br>
The required dependencies are managed using poetry. <br>

## <a id='installation'></a> Installation:
To install the requirements:

```
poetry install
```

To start the virtual environment created by poetry:

```
poetry shell
```

---

### <a id='example'></a> Example of application in a multimodal setting:
Reindl, V., Wass, S., Leong, V., Scharke, W., Wistuba, S., Wirth, C. L., ... & Gerloff, C. (2022). Multimodal hyperscanning reveals that synchrony of body and mind are distinct in mother-child dyads. NeuroImage, 251, 118982, [10.1016/j.neuroimage.2022.118982](https://doi.org/10.1016/j.neuroimage.2022.118982).
## <a id='references'></a>References:

If you use the code, please cite:

```
Gerloff C, Konrad K, Bzdok D, Büsing C, Reindl V. Interacting brains revisited: A cross-brain network neuroscience perspective. Human Brain Mapping. 2022 Jun 6. doi: 10.1002/hbm.25966.
```


If you have any questions or comments, or if you are interested in collaborating with us, feel free to [contact me](mailto:christian.gerloff@rwth-aachen.de).