# Feature matching using SIFT

This folder explores how we can register retinal images using SIFT feature detection and descriptor extraction + descriptors matching + ransac.

## How to run

- Place yourself into the `ml4s-py3.8` environment.
- Run the .py file using `python3 feature_matching.py`.

At the top of this file, you can specify the input dataset and output folders. Algorithms can also be adapted.

By default, registered images will be placed into the `full_output` folder, and grouped into folders by patient, eye and centerness.

## Notebook

- The `feature_matching` notebook explains the chosen pipeline and plot some examples.
- The `registration_evaluation` notebook contains an analysis of the results of the registration.

## Results

Results can be found in the `full_output` folder.

Complete results (including several metrics I guess) will be given in the paper and are also in the `registration_evaluation` notebook. Nevertheless, exploring the registered images, we feel satisfied of the results we can see for most of the cases (even if some registrations are incorrects).

Examples that don't work are 102-R-LQ for example, and we can think it's because vessels are very hard to see. Finding the algorithms parameters is okay for most images, but one adapted to all of them is way harder.

The next step might be to explore the hyperparameters even more, maybe as first through grid search or other techniques.
