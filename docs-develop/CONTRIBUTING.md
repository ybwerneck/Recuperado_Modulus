# Contributing to Modulus Documentation


# Table of Contents

<!-- toc -->
- [Modulus User Guide Examples](#modulus-user-guide-examples)
- [Modulus Doc Strings](#modulus-doc-strings)
<!-- tocstop -->

## Mission: Clarity and Consistency

# Modulus User Guide Examples

## Examples Template

## User Guide Spell Checker

Modulus documentation has a basic spell-checking CI test for all user guide `.rst` files.
The spellchecker uses a base English dictionary as well as an extended dictionary with technical domain-specific terms.
The spell-checker is case invariant and is not context aware.
If your CI tests fail this may likely be the reason, check the pipelines page and the **test spelling** job.
Some parts of RST files are not considered in the spell checker including code blocks, math blocks, figures, tables, inline single/double quote blocks, links, citations, and references.

### Spelling Errors

An error will be thrown if the spell checker finds a word that is within [2 Levenshtein edit distance](https://pyspellchecker.readthedocs.io/en/latest/) from a word in the dictionary.
This could be because of a silly mistake, adding an unnecessary hyphen or because this word is not present in the dictionary.
You have a few options options: 1) correct the word or change it, 2) if text is code/math make it an Sphinx inline code/math block or 3) add the word to the Modulus dictionary.

### Spelling Warnings

The spell checker will give a warning if it sees a word that is unlike any present in the dictionary.
This allows for new names or technical jargon to be added without preventing a merge.
However, it is *highly* encouraged you correct these warnings by either rewording or adding this work to the Modulus dictionary.

### Adding Words to the Dictionary

When a new word is *absolutely* needed, add it to the Modulus dictionary `test/modulus_dictionary.json`.
Simply add it to the dictionary list inside the JSON and it will then be considered in the CI test.
The addition of terms into the dictionary should be kept to a minimum and will be more extensively reviewed.

If the needed word is a part of the Sphinx RST language (e.g. `:code:`, `.. math::`, `.. figure::`, etc), contact Nicholas Geneva to get the CI test adjusted with additional regex parsing rules.
Do not add Sphinx related text to the dictionary.


# Modulus Doc Strings

Properly documenting Modulus API is one of the most important parts of development, since this directly communicates with users on how to use various features in Modulus.
In an effort to help create clarity and consistency for all API documentation, Modulus has a set template one should always follow for Python docstrings.
Modulus' docstrings are based on the [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html) but with some additional rules.

## Docstring Standards

In general, the following section order should be used for all doc strings.
Not every section needs to be present, it is the order that is important.
Each section headers should be denoted by a underline of minus signs `-----` that are the **exact** width of the title, separated by one blank line and content for a section should start on the same indentation level.
Doc stings should have a **max-linewidth of 88 characters** (following black format standards), with the exception of example python code.

### Classes
Classes are more likely to be documented than not. If an end user may initialize this class inside of a script, then it should be documented.
Doc strings for classes should be placed *before* the constructor method. 
**Constructors should never have docstrings**, class parameters should be included in the preceding docstring.

1. *Short Summary* - A one-line brief summary of what the class is. Should be on the same line as the initial `"""`.

2. *Extended Summary* - An optional longer summary of the class and what its functionality is. This may also entail the nature of its use cases. This does NOT include implementation details. The description should be seperated from the short discription by a blank line.

3. *Use Notes* - Notes that are critical to the usage of the class should be listed. This may be limitation notes or key features that are likely to cause confusion. These are different than reference notes at the end of the doc string.
```python
    """
    Note
    ----
    AFNO is a model that is designed for 2D images only.
    """
```

4. *Parameters* - This section should be used to describe the parameters of the classes constructor. When beneficial this may include the parameters of parent classes. All parameters should have a type cast. For default parameters `, optional` should be added after the type.
```python
    """
    Parameters
    ----------
    input_keys : List[Key]
        Input key list. The key dimension size should equal the variables channel dim.
    output_keys : List[Key]
        Output key list. The key dimension size should equal the variables channel dim.
    dimension : int
        Model dimensionality (supports 1, 2, 3).
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    """
```
After the name of the parameter and the type, the following line should be a brief description of what this parameter is.
Complete sentences should be punctuated.
At the end of the description, default values should then be specified via `, by default <value>` or `By default <value>` with no punctuation.

5. *Attributes* - If the class has important attributes/properties that the user will likely need to access they should be documented in this section. 
Private attributes denoted with a leading underscore should never be documented. 
```python
    """
    Attributes
    ----------
    weight : torch.Tensor
        Weight tensor of fully-connected layer.
    """
```

6. *Variable Shape* - This section is one that is unique to Modulus to help document the input/output variable shapes for a particular model or function. This should primarily be used for neural network architectures.
```python
    """
    Variable Shape
    --------------
    - Input variable tensor shape: :math:`[N, size, H, W]`
    - Output variable tensor shape: :math:`[N, size, H, W]`
    """
```

This section should include two bullets, the first being `Input variable tensor shape` and the next being `Output variable tensor shape`. The shapes are the dimensionality of the tensor with square brackets, `N` should be used for batch dim, `size` used for variable dimension (or size in Key()), and remaining dims can be named appropriately.

7. *Example(s)* - Examples should always be included when possible for classes. A good example is no more than 10 lines of code to show a classes initialization and basic use.
An example does not need to show all functionality or parameters, but rather a fundamental use case. Import statements should be excluded when possible.
```python
    """
    Example
    --------
    >>> afno = modulus.architecture.afno.AFNOArch([Key("x", size=2)], [Key("y", size=2)], (64, 64))
    >>> model = afno.make_node()
    >>> input = {"x": torch.randn(20, 2, 64, 64)}
    >>> output = model.evaluate(input)
    """
```

Python code should be denoted with `>>>`. 
When not obvious inline comments can be used to tell users details about the output size or type.

8. *Reference Notes* - Additional note sections can be placed after the example to discuss less critical details that users/developers may need to be aware of.
Examples of information that should be added to this section is plans for future changes, implementation details, references to other parts of the code, paper references etc.


### Functions and Methods
For functions or methods that are essential for users, docstrings should be added. Otherwise, docstring should not be used in favor of regular comments for developers. This is to help keep the API documentation focused on user-facing methods.

1. *Short Summary* - A one-line brief summary of what the function/method is. Should be on the same line as the initial `"""`.

2. *Extended Summary* - An optional longer summary of the function. This does NOT include implementation details. The description should be seperated from the short discription by a blank line.
```python
def subtract(a, b):
    """Subtraction of two numbers.
    
    Subtracts the two numbers a and b, which is a basic mathermatical function that does
    not really serve any purpose.
    """
```

3. *Use Notes* - Notes that are critical to the usage of the method/function should be listed above the parameters. Details that are not showstoppers should be added in the reference notes section below.

4. *Parameters* - This section should be used to describe the parameters of the method/function. All parameters should have a type cast. For default parameters `, optional` should be added after the type.
```python
    """
    Parameters
    ----------
    bounds : List[List[int]]
        Domain bounds of each dimension
    npoints : List[int]
        List of number of points in each dimension
    export_map : Dict[str, List[str]], optional
        Export map dictionary with keys that are VTK variables names and values that are 
        lists of output variables. Will use 1 to 1 mapping if none is provided, by 
        default {}
    file_name : str, optional
        File name of output vtk file, by default "vtk_output"
    """
```
After the name of the parameter and the type, the following line should be a brief description of what this parameter is.
Complete sentences should be punctuated.
At the end of the description, default values should then be specified via `, by default <value>` or `By default <value>` with no punctuation.

5. *Returns* - Description of the return values. Returns should not be named, only a type and description. Multiple outputs can be specified similar to the parameters.
```python
    """
    Returns
    -------
    loss : torch.Tensor
        Aggregated loss
    """
```

6. *Raises* - Errors that a user may encounter during use under certain conditions. This section should be used only for non-obvious errors with error messages that may confuse the user or common failures.

7. *Example(s)* - Examples for functions typically more important than examples for class methods.
```python
    """
    Example
    -------
    >>> afno = modulus.architecture.afno.AFNOArch([Key("x", size=2)], [Key("y", size=2)], (64, 64))
    >>> model = afno.make_node()
    >>> input = {"x": torch.randn(20, 2, 64, 64)}
    >>> output = model.evaluate(input)
    """
```

Python code should be denoted with `>>>`. 
When not functionality is not obvious, inline comments can be used to tell users details about the output size or type.

8. *Reference Notes* - Additional note sections can be placed after the example to discuss less critical details that users/developers may need to be aware of.
Examples of information that should be added to this section are plans for future changes, implementation details, references to other parts of the code, paper references etc. 


## Examples

```python
class AFNOArch(Arch):
    """Adaptive Fourier neural operator (AFNO) model.

    Note
    ----
    AFNO is a model that is designed for 2D images only.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list. The key dimension size should equal the variables channel dim.
    output_keys : List[Key]
        Output key list. The key dimension size should equal the variables channel dim.
    img_shape : Tuple[int, int]
        Input image dimensions (height, width)
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    patch_size : int, optional
        Size of image patchs, by default 16
    embed_dim : int, optional
        Embedded channel size, by default 256
    depth : int, optional
        Number of AFNO layers, by default 4
    num_blocks : int, optional
        Number of blocks in the frequency weight matrices, by default 4


    Variable Shape
    --------------
    - Input variable tensor shape: :math:`[N, size, H, W]`
    - Output variable tensor shape: :math:`[N, size, H, W]`

    Example
    -------
    >>> afno = modulus.architecture.afno.AFNOArch([Key("x", size=2)], [Key("y", size=2)], (64, 64))
    >>> model = afno.make_node()
    >>> input = {"x": torch.randn(20, 2, 64, 64)}
    >>> output = model.evaluate(input)
"""
```

```python
class VTKPolyData(VTKBase):
    """vtkPolyData wrapper class

    Parameters
    ----------
    points : np.array
        Array of point locations [npoints, (1,2 or 3)]
    line_index : np.array, optional
        Array of line connections [nedges, 2], by default None
    poly_index : Tuple[poly_offsets, poly_connectivity]
        Tuple of polygon offsets and polygon connectivity arrays.
        Polygon offsets is a 1D array denoting how many points make up a face for each 
        polygon. Polygon connectivity is a 1D array that contains verticies of each 
        polygon face  in order, by default None
    export_map : Dict[str, List[str]], optional
        Export map dictionary with keys that are VTK variables names and values that are 
        lists of output variables. Will use 1 to 1 mapping if none is provided, by 
        default {}
    file_name : str, optional
        File name of output vtk file, by default "vtk_output"
    file_dir : str, optional
        File directory of output vtk file, by default "."
    init_vtk : bool, optional
        Initialize new VTK object from parameters (used by VTKFromFile), by default True
    """
```

```python
def get_stencil_input(
        self, inputs: Dict[str, Tensor], stencil_strs: List[str]
    ) -> Dict[str, Tensor]:
    """Creates a copy of the inputs tensor and adjusts its values based on stencil str.

    Parameters
    ----------
    inputs : Dict[str, Tensor]
        Input tensor dictionary
    stencil_strs : List[str]
        batch list of stencil string from derivative class

    Returns
    -------
    Dict[str, Tensor]
        Modified input tensor dictionary

    Example
    -------
    A stencil string `x::1` will modify inputs['x'] = inputs['x'] + dx
    A stencil string `y::-1,z::1` will modify inputs['y'] = inputs['y'] - dx and
    inputs['z'] = inputs['z'] + dx
    """
```

## Tools
There are various tools that can help template/draft docstrings, here are a few of the suggested ones.
Note that no tool is perfectly aligned with Modulus' standards, thus the user should always verify the documentation themselves.

### VSCode - autoDocstring

The [autoDocstring extension](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/sphinxcontrib.napoleon.html#module-sphinxcontrib.napoleon.docstring) for VSCode can be used to quickly generated boilerplate code for python docstrings. Simply install the extension then switch to the numpy format by:

- File > Preferences > Settings
- Search "docstring"
- Fine "Auto Docstring: Docstring Format"
- Select numpy from the drop-down

To use, simple type `"""` under a function or class and press enter when prompted to create a docstring.
Additionally [vertical rulers](https://stackoverflow.com/a/29972073) can be added into VSCode to denote the 88 character limit.

### pydocstyle

[pydocstyle](http://www.pydocstyle.org/en/stable/index.html) can be used to help verify a file's dock string which is installed by default in the documentation docker image.
Modulus docs has a custom `.pydocstyle.ini` file to use in the root of the repo.
```bash
pydocstyle ./modulus/SimNet/modulus/architecture/fno.py
```

## References

- [NumPy docstring style guide](https://numpydoc.readthedocs.io/en/latest/format.html) 
- [NumPy docstring examples](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html)