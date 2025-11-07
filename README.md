# GCOS Reference Upper Air Network (GRUAN) Exploratory Data Analysis (EDA) by GRUANpy

This repo is meant to collect python developed functionalities to analyze GRUAN Data Product.

## What is GRUAN?

"The Global Climate Observing System (GCOS) Reference Upper-Air Network (GRUAN) is an international reference observing network of sites measuring essential climate variables above Earth's surface, designed to fill an important gap in the current global observing system. GRUAN measurements are providing long-term, high-quality climate data records from the surface, through the troposphere, and into the stratosphere. These are being used to determine trends, constrain and calibrate data from more spatially‐comprehensive observing systems (including satellites and current radiosonde networks), and provide appropriate data for studying atmospheric processes. GRUAN is envisaged as a global network of eventually 30-40 sites that, to the extent possible, builds on existing observational networks and capabilities." 
https://www.gruan.org/

## What are GRUAN Data Products (GDP)?

GDP are documented NetCDF files containing data and metadata from GRUAN.
"All certified GRUAN data products are based on measurements and processing that adhere to the GRUAN principles (e.g. [Immler et al., 2010](https://www.gruan.org/documentation/articles/immler-et-al-2010-amt))."

https://www.gruan.org/data/data-products/gdp

Certified GRUAN data products
- RS92-GDP.2 -- RS92 GRUAN data product version 2
- RS-11G-GDP.1 -- RS-11G GRUAN data product version 1
- RS41-GDP.1 -- RS41 GRUAN data product version 1
- IMS-100-GDP.2 -- iMS-100 GRUAN data product version 2

## What is GRUANpy?

GRUANpy is a toolkit for analizing GDP. Hope to pubblish it on PyPI at some point. 🤞

## How to use it?

Just put in the "gdp" folder (or wherever you like) a GDP.nc file and import gruanpy to:
- read it
- make fancy plots
- compute PBLH
- ...

The limit is the sky, literally.

You can find some example inside the "code_example" folder.

## Some tips.

NetCDF Explorer (VSCode Extension) to let us look at .nc.info files.