# Introduction to Machine Learning

### [View Presentation](http://donaldwhyte.github.io/intro-to-ml/automated)

Presentation briefly introducing machine learning, what is it, why its use has grown significantly in the past decade, its current state and how to apply it to solve real world problems.

Topics covered:

* machine learning definition
* growth of adoption
* supervised learning
    - classification
    - regression
    - feature extraction
    - model types and overfitting
    - evaluation (with k-fold cross validation)
* tools
    - Python
    - scikit-learn examples
* faces vs. vegetables
    - example classification example
* process of applying machine learning
* how to automate a lot of the process

## Running Presentation

You can also run the presentation on a local web server. Clone this repository and run the presentation like so:

```
npm install
grunt serve
```

The presentation can now be accessed on `localhost:8080`. Note that this web application is configured to bind to hostname `0.0.0.0`, which means that once the Grunt server is running, it will be accessible from external hosts as well (using the current host's public IP address).

## Executing Demos

```
cd demo

# Install dependencies in new own virtualenv
virtualenv env
source env/bin/activate
pip install -r requirements.txt

# Run examples
./spotcheck_facesvegetables.py
./automl_digits.py
./automl_facesvegetables.py
```
