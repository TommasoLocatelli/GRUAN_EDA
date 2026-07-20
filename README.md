# GCOS Reference Upper-Air Network (GRUAN) Exploratory Data Analysis (EDA) with **GRUANpy**

A collection of Python tools designed to analyze GRUAN Data Products (GDPs) and support research on atmospheric profiles, uncertainty quantification, and boundary-layer diagnostics.

---

## What is GRUAN?

> “The Global Climate Observing System (GCOS) Reference Upper-Air Network (GRUAN) is an international reference observing network of sites measuring essential climate variables above Earth's surface, designed to fill an important gap in the current global observing system. GRUAN measurements provide long-term, high-quality climate data records from the surface, through the troposphere, and into the stratosphere. GRUAN is envisaged as a global network of eventually 30–40 sites that builds on existing observational networks and capabilities.”  
> — https://www.gruan.org/

---

## What are GRUAN Data Products (GDPs)?

GRUAN Data Products are documented NetCDF files containing high‑quality atmospheric measurements and metadata.  
All certified GDPs follow GRUAN principles (e.g., [Immler et al., 2010](https://www.gruan.org/documentation/articles/immler-et-al-2010-amt)).

More info: https://www.gruan.org/data/data-products/gdp

---

## What is **GRUANpy**?

**GRUANpy** is a Python toolkit for reading, analyzing, and visualizing GRUAN Data Products.  
It includes utilities for:

- loading GDP NetCDF files  
- generating diagnostic and exploratory plots  
- computing Planetary Boundary Layer Height (PBLH)  
- performing state‑space modelling (SSM) and uncertainty analysis  
- supporting reproducible workflows for atmospheric research  

The long‑term goal is to publish GRUANpy on PyPI. 🤞

---

## How to Use It

Place one or more `*.nc` GDP files in the `gdp/` folder (or any directory you prefer) and import `gruanpy` in your Python script or notebook to:

- read and inspect GDP variables  
- create plots of atmospheric profiles  
- compute PBLH using multiple diagnostics  
- run SSM-based smoothing and uncertainty quantification  
- explore radiosonde data with reproducible workflows  

Examples are available in the `applications/` folder, including the code supporting the paper:  
**https://arxiv.org/abs/2607.14960**

---

## Repository Structure

- **applications/** — example scripts and workflows  
  - **applications/pblh_unc/** — script supporting the publication at http://arxiv.org/abs/2607.14960
- **gdp/** — directory intended to store GDP NetCDF files  
- **gruanpy/** — core library with GDP utilities and diagnostics  
- **ssm/** — state‑space modelling and Kalman smoothing tools

---

## Tips

If you use VS Code, the **NetCDF Explorer** extension is extremely helpful for inspecting `.nc` files and metadata.
