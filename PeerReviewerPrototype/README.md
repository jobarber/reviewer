# PeerReviewerPrototype
This is the repository that will hold the complete peer review prototype slated to be completed by December 31, 2021. I have required pull requests for this repo as well as automated testing for continuous integration. I will add code here at no more than 100 lines at a time, and I will need one of you (Michael and Philip) to review the code, make comments, and approve it for each pull request. The idea here is twofold: (1) check my implementation as best you can to see if it makes sense in terms of our broad goals and (2) learn the repository yourselves so that you can work or build upon it in the future.

If you work off your own branches, please create branches with the following format:

  <github handle>-<short summary of branch>

For example, I am naming this branch as follows:
  
  jobarber-AddingInstructionsToReadMe

You can run inference on this package with the following commandline arguments when run from the PeerReviewerPrototype
directory. The academic theology model will be downloaded the first time if you do not have it.

First, you will want to install the necessary requirements for your Python interpreter:

```
PeerReviewerPrototype$ pip3 install -r requirements.txt
```

Then you should be able to run the main module on a doc of your choice.

```
$ python main.py --help

usage: main.py [-h] --path_to_docx PATH_TO_DOCX [--path_to_output PATH_TO_OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  --path_to_docx PATH_TO_DOCX
                        the path to your docx file
  --path_to_output PATH_TO_OUTPUT
                        where to place the output file

```

For example:

```
$ python main.py --path_to_docx /home/iliff/Downloads/Sample\ Exegetical\ Paper\ 2019.docx --path_to_output /home/iliff/REVIEWED\ Sample\ Exegetical\ Paper\ 2019.docx
```


