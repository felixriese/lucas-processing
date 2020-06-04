.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3871431.svg
   :target: https://doi.org/10.5281/zenodo.3871431
   :alt: Zenodo

LUCAS Soil Texture Processing Scripts
=====================================

This repository is a placeholder for the processing scripts of the soil
texture data included into the LUCAS dataset [5]. This script is used for a
study [3, 4] of 1D Convolutional Neural Networks (CNNs). The code of the 1D
CNNs is published in [2].

We can not guarantee completeness or correctness of the code. If you find bugs
or if you have suggestions on how to improve the code, we encourage you to post
your ideas as `GitHub issue
<https://github.com/felixriese/lucas-processing/issues>`_.

:License:
    `3-Clause BSD license <LICENSE>`_

:Author:
    `Felix M. Riese <mailto:github@felixriese.de>`_

:Requirements:
    Python 3 with these `packages <requirements.txt>`_

:Citation:
    see `Citation`_ and in the `bibtex <bibliography.bib>`_ file

Notebooks
---------

1. `Process LUCAS dataset <py/Process_LUCAS_Dataset.ipyn>`_
2. `Plot soil triangle <py/Plot_SoilTriangle.ipyn>`_
3. `PLot results <py/Plot_Results.ipyn>`_

----

Citation
--------

**Code for the Scripts:**

[1] F. M. Riese, "LUCAS Soil Texture Processing Scripts," Zenodo, 2020.
`DOI:0.5281/zenodo.3871431 <https://doi.org/10.5281/zenodo.3871431>`_

.. code:: bibtex

    @misc{riese2020lucas,
        author = {Riese, Felix~M.},
        title = {{LUCAS Soil Texture Processing Scripts}},
        year = {2020},
        DOI = {10.5281/zenodo.3871431},
        publisher = {Zenodo}
    }

**Code for the 1D CNNs:**

[2] F. M. Riese, "CNN Soil Texture Classification", Zenodo, 2019.
`DOI:10.5281/zenodo.2540718 <https://doi.org/10.5281/zenodo.2540718>`_

.. code:: bibtex

    @misc{riese2019cnn,
        author = {Riese, Felix~M.},
        title = {{CNN Soil Texture Classification}},
        year = {2019},
        DOI = {10.5281/zenodo.2540718},
        publisher = {Zenodo},
    }

Code is Supplementary Material to
---------------------------------

[3] F. M. Riese and S. Keller, "Soil Texture Classification with 1D
Convolutional Neural Networks based on Hyperspectral Data", ISPRS Annals of
Photogrammetry, Remote Sensing and Spatial Information Sciences, vol. IV-2/W5,
pp. 615-621, 2019. `DOI:10.5194/isprs-annals-IV-2-W5-615-2019
<https://doi.org/10.5194/isprs-annals-IV-2-W5-615-2019>`_

[4] Felix M. Riese. "Development and Applications of Machine Learning Methods
for Hyperspectral Data." PhD thesis. Karlsruhe, Germany: Karlsruhe Institute of
Technology (KIT), 2020.

Further References
------------------

[5] G. Tóth, A. Jones, and L. Montanarella, "LUCAS Topsoil Survey: Methodology,
Data, and Results." Tech. rep. JRC83529. Joint Research Centre of the European
Commission, 2013. `DOI:10.2788/97922 <https://doi.org/10.2788/97922>`_
