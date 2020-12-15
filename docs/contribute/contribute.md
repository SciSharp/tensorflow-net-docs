# Contribute

Feel like contributing to one of the hottest projects in the Machine Learning field? Want to know how TensorFlow magically creates the computational graph? We appreciate every contribution however small. There are tasks for novices to experts alike, if everyone tackles only a small task the sum of contributions will be huge.

You can:
* Let everyone know about this project
* Port TensorFlow unit tests from Python to C# or F#
* Port missing TensorFlow code from Python to C# or F#
* Port TensorFlow examples to C# or F# and raise issues if you come accross missing parts of the API
* Debug one of the unit tests that is marked as Ignored to get it to work
* Debug one of the not yet working examples and get it to work

### How to debug unit tests

The best way to find out why a unit test is failing is to single step it in C# or F# and its corresponding Python at the same time to see where the flow of execution digresses or where variables exhibit different values. Good Python IDEs like PyCharm let you single step into the tensorflow library code. 

### Git Knowhow for Contributors

Add SciSharp/TensorFlow.NET as upstream to your local repo ...
```bash
git remote add upstream git@github.com:SciSharp/TensorFlow.NET.git
```

Please make sure you keep your fork up to date by regularly pulling from upstream. 
```bash
git pull upstream master
```
