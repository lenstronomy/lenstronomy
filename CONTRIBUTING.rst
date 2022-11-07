Contributor Guidelines
======================

GitHub Workflow
---------------

Fork and Clone the lenstronomy Repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**You should only need to do this step once**

First *fork* the lenstronomy repository. A fork is your own remote copy of the repository on GitHub. To create a fork:

  1. Go to the `lenstronomy GitHub Repository <https://github.com/lenstronomy/lenstronomy>`_
  2. Click the **Fork** button (in the top-right-hand corner)
  3. Choose where to create the fork, typically your personal GitHub account

Next *clone* your fork. Cloning creates a local copy of the repository on your computer to work with. To clone your fork:

::

   git clone https://github.com/<your-account>/lenstronomy.git


Finally add the ``lenstronomyproject`` repository as a *remote*. This will allow you to fetch changes made to the codebase. To add the ``lenstronomyproject`` remote:

::

  cd lenstronomy
  git remote add lenstronomyproject https://github.com/lenstronomy/lenstronomy.git



Install your local lenstronomy version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To enable that your new code gets accessible by python also outside of the development environment, 
make sure all previous versions of lenstronomy are uninstalled and then install your version of lenstronomy (aka add the software to the python path)

::

  cd lenstronomy
  python setup.py develop --user


Alternatively, create virtual environments for the development (recommended for advanced usage with multiple branches).



Create a branch for your new feature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a *branch* off the `lenstronomyproject` main branch. Working on unique branches for each new feature simplifies the development, review and merge processes by maintaining logical separation. To create a feature branch:

::

  git fetch lenstronomyproject
  git checkout -b <your-branch-name> lenstronomyproject/main


Hack away!
^^^^^^^^^^

Write the new code you would like to contribute and *commit* it to the feature branch on your local repository. Ideally commit small units of work often with clear and descriptive commit messages describing the changes you made. To commit changes to a file:

::

  git add file_containing_your_contribution
  git commit -m 'Your clear and descriptive commit message'


*Push* the contributions in your feature branch to your remote fork on GitHub:

::

    git push origin <your-branch-name>



**Note:** The first time you *push* a feature branch you will probably need to use `--set-upstream origin` to link to your remote fork:

  
::

  git push --set-upstream origin <your-branch-name>


Open a Pull Request
^^^^^^^^^^^^^^^^^^^

When you feel that work on your new feature is complete, you should create a *Pull Request*. This will propose your work to be merged into the main lenstronomy repository.

  1. Go to `lenstronomy Pull Requests <https://github.com/lenstronomy/lenstronomy/pulls>`_
  2. Click the green **New pull request** button
  3. Click **compare across forks**
  4. Confirm that the base fork is ``lenstronomy/lenstronomy`` and the base branch is ``main``
  5. Confirm the head fork is ``<your-account>/lenstronomy`` and the compare branch is ``<your-branch-name>``
  6. Give your pull request a title and fill out the the template for the description
  7. Click the green **Create pull request** button

Updating your branch
^^^^^^^^^^^^^^^^^^^^

As you work on your feature, new commits might be made to the ``lenstronomy/lenstronomy`` main branch. You will need to update your branch with these new commits before your pull request can be accepted. You can achieve this in a few different ways:

  - If your pull request has no conflicts, click **Update branch**
  - If your pull request has conflicts, click **Resolve conflicts**, manually resolve the conflicts and click **Mark as resolved**
  - *merge* the ``lenstronomyproject`` main branch from the command line:
    ::

        git fetch lenstronomyproject
        git merge lenstronomyproject/main


  - *rebase* your feature branch onto the ``lenstronomy`` main branch from the command line:

    ::

        git fetch lenstronomyproject
        git rebase lenstronomyproject/main


**Warning**: It is bad practice to *rebase* commits that have already been pushed to a remote such as your fork.
Rebasing creates new copies of your commits that can cause the local and remote branches to diverge. ``git push --force`` will **overwrite** the remote branch with your newly rebased local branch.
This is strongly discouraged, particularly when working on a shared branch where you could erase a collaborators commits.

For more information about resolving conflicts see the GitHub guides:
  - `Resolving a merge conflict on GitHub <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/resolving-a-merge-conflict-on-github>`_
  - `Resolving a merge conflict using the command line <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/resolving-a-merge-conflict-using-the-command-line>`_
  - `About Git rebase <https://help.github.com/en/github/using-git/about-git-rebase>`_

More Information
^^^^^^^^^^^^^^^^

More information regarding the usage of GitHub can be found in the `GitHub Guides <https://guides.github.com/>`_.

Coding Guidelines
-----------------

Before your pull request can be merged into the codebase, it will be reviewed by one of the lenstronomy developers and required to pass a number of automated checks. Below are a minimum set of guidelines for developers to follow:

General Guidelines
^^^^^^^^^^^^^^^^^^

- lenstronomy is compatible with Python>=3.7 (see `setup.cfg <https://github.com/lenstronomy/lenstronomy/blob/main/setup.cfg>`_). lenstronomy *does not* support backwards compatibility with Python 2.x; `six`, `__future__` and `2to3` should not be used.
- All contributions should follow the `PEP8 Style Guide for Python Code <https://www.python.org/dev/peps/pep-0008/>`_. We recommend using `flake8 <https://flake8.pycqa.org/>`__ to check your code for PEP8 compliance.
- Importing lenstronomy should only depend on having `NumPy <https://www.numpy.org>`_, `SciPy <https://www.scipy.org/>`_ and `Astropy <https://www.astropy.org/>`__ installed.
- Code is grouped into submodules based e.g. `LensModel <https://lenstronomy.readthedocs.io/en/latest/lenstronomy.LensModel.html>`_, `LightModel <https://lenstronomy.readthedocs.io/en/stable/lenstronomy.LightModel.html>`_ or  `ImSim <https://lenstronomy.readthedocs.io/en/latest/lenstronomy.ImSim.html>`_. There is also a `Util <https://lenstronomy.readthedocs.io/en/stable/lenstronomy/Util.html>`_ submodule for general utility functions.
- For more information see the `Astropy Coding Guidelines <http://docs.astropy.org/en/latest/development/codeguide.html>`_.


Unit Tests
^^^^^^^^^^

Pull requests will require existing unit tests to pass before they can be merged.
Additionally, new unit tests should be written for all new public methods and functions.
Unit tests for each submodule are contained in subdirectories called ``tests`` and you can run them locally using ``python setup.py test``.
For more information see the `Astropy Testing Guidelines <https://docs.astropy.org/en/stable/development/testguide.html>`_.

Docstrings
^^^^^^^^^^

All public classes, methods and functions require docstrings. You can build documentation locally by installing sphinx and calling ``python setup.py build_docs``. Docstrings should include the following sections:

  - Description
  - Parameters
  - Notes
  - Examples
  - References

For more information see the Astropy guide to `Writing Documentation <https://docs.astropy.org/en/stable/development/docguide.html>`_.

This page is inspired by the Contributions guidelines of the `Skypy project <https://github.com/skypyproject/skypy/blob/main/CONTRIBUTING.rst>`_.
