# Steps to install and run our ENsom model.
1. Download/clone the mypackage folder locally .
2. Open the mypackage/dist/ folder and type the below command. This installs the required class with the respective functions in our distro.
```
  #pip3 install ENsom-0.1.tar.gz .
```
3. Then you can easily open any python file and import the class from it.
```
  # import the class with all its functions
  from ENsom.som_class import *
  
  # create an object with the given/user defined parameters
  obj = SOM_EN(7,7,1,100,0.5)
  
  # load the dataset into the object
  obj.load_dataset(r"add the path to the csv file",False)
  
  # tuning the som model
  obj.tuning()
 
  # train and plot the model
  obj.train_som()
  obj.plot_som()
```
