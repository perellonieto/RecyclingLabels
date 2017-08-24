# RecyclingLabels

[![Build Status](https://travis-ci.com/perellonieto/RecyclingLabels.svg?token=bCq7XPyjnZso4MsN7scu&branch=master)](https://travis-ci.com/perellonieto/RecyclingLabels)

Code that shows how to reuse old labels that are known to contain errors (weak
labels) by adding knowledge about how a sub-sample of this labels has evolved
over time by the acquisition of new true labels

### Unittest

```bash
./runtests.sh
```

### Installation

```bash
git clone git@github.com:perellonieto/RecyclingLabels.git
cd RecyclingLabels
virtualenv venv
. venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Tutorial

The script `tutorial.py` runs an example of the Recycling Labels approach on
the specified dataset from the options iris, blobs and webs. It generates a
markdown. For example, to run the tutorial on iris dataset execute:

```bash
python tutorial.py iris
```

The generated output can be rendered to html for example using pandoc. For
example:

```bash
python tutorial.py iris > tutorial_iris.markdown
pandoc tutorial_iris.markdown > tutorial_iris.html
```
