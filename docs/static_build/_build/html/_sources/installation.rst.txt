============
Installation
============

Sam requires the following libraries to compile:

* Boost -- specifically the Random, Special, and Math libraries.
* The Python dev libraries (e.g. python-dev or python3-dev)
* Cython
* Numpy
* Scipy
* Multiprocessing

Once these are installed, you can compile Sam with the following command from the Sam directory:

.. code-block:: bash

    pip install --user .

If you prefer not to use pip, you may instead use:

.. code-block:: bash

    python setup.py install --user

Finally, for a system-wide install you may omit ``--user``, although this may require elevated user privileges.
