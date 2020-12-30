# PySparseTool

Package that perform spsification of POD-Galerkin models of turbulent flows. 

## Dependencies

This module relies on the sklearn implementation of the LASSO regression. 
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install sklearn and every other needed dependencies.

```bash
pip3 install sklearn
```

## Usage

Given the modal amplitude and the modal acceleration the user needs to perform the following steps:

1) Generate the database matrix containing the constant linear and quadratic interactions

```bash

python3 Generate_Dataset.py N

```
with N the number of modes the user wants to be considered in the reduced order model.

2) Perform sparsification. Due the convenient mathematical structure is recommended to solve the LASSO problem in parallel with the command

```bash

mpiexec -n Np python3 LASSO.py            

```
with Np the number of processors.

3) Once the problem is solved we can reconstruct the solution with 

```bash

python3 prepare_data_for_plot.py 
     
```
the solution is saved in the plotting_data folder for some post processing.

<!-- 
```python
import foobar
foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```
-->

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
