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
