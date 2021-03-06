# Running the Code

## Tech Stack
 
1) Python 3.8+
2) TensorFlow v2.5+
3) Install packages listed in the ``` requirements.txt ``` file.
4) For Windows, refer to the following <a href="https://devblogs.microsoft.com/scripting/table-of-basic-powershell-commands/" target="_blank"> Windows Commands Website </a>
5) For Mac, refere to <a href="https://support.apple.com/guide/terminal/execute-commands-and-run-tools-apdb66b5242-0d18-49fc-9c47-a2498b7c91d5/2.9/mac/10.14" target="_blank"> the Mac commands Page </a>
6) The instructions described on this Page were performed on Linux.

<hr>

##  Installation requirements
<hr>

- After cloning the repository, change to the main folder

       cd ~/apin-fs-paper/apin-fs/

where ``` ~ ``` represents the home directory.

- Install the Packages listed the ``` requirements.txt ``` file as follows

      pip install -r requirements.txt


In case you decide to install the contents of the requirements manually (in situations where there are issues regarding version conflicts), install each package individually. For example, <a href="https://www.tensorflow.org/" target="_blank">TensorFlow .2.0 from the Website</a>

<hr>

### Recommended Packages for Running Code
<hr>


Use either the  Anaconda Python 3.7+ or the venv provided in Python to create a new environment. Details can be found here and also on this Website. From the <a href="https://docs.anaconda.com/anaconda/packages/py3.9_win-64/" target="_blank"> Anaconda Website </a> and the ``` venv ``` is contained in Python.
### Instructions on how run the code:


1) Change to the source code directory containing the main training file.
<hr>


- Change to the folder containing the code

      cd ~/apin-fs-paper/apin-fs/src/models

Assuming the correct versions of all packages in the requirements file have been installed, run the following code to train the model

     python train_model.py

To visualize the Performance statitics for the Training and Testing data sets for different Algorithms:

For the heart disease data set:

     cd ~/apin-fs-paper/apin-fs/reports/h_data/
     python h_data-results.py

To visualize performance on the WCDS data set

     cd ~/apin-fs-paper/apin-fs/reports/wcds_data/
     python wcds_results.py

For the Heart Disease data:

     python ~/apin-fs-paper/apin-fs/reports/h_data/h_data-results.py
To run notebooks for analysis and other vital statistics about the data sets

     cd ~/apin-fs-paper/apin-fs/notebooks

Data set folders:

     cd ~/apin-fs-paper/apin-fs/src/data/

Change the choices for different data sets. For choices 1 and 3, following data files are currently available:

The Artificial Data set for regression:

    cd raw/testData.mat

WCDS data set for classification:

     cd raw/data.csv

Change the choices for different data sets. For choices 1 and 3 ( for WCDS represented by ```data.csv``` and ```testData.mat``` respectively)

The ``` data ``` folder will continuously be updated with other data sets.

Note: This page will be continuously be updated (such as adding files for the ouput during training). Continuous Updates are required due to the continuous change in the technology needed to run the code. kindly bear with us. Meanwhile, the authorized usage of the code is available to the APIN Journal. For other request, authorization or duplication, please contact the owner via the following email: <nathaniel@aims.ac.za> or <message4nath@gmail.com>.
