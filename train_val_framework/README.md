##The Framework

1. Folder Structure  
 - All files are in **train\_val\_framework** folder. 
 - **test_framework.py** should be the main function where there will be a lot of parameters you could enter (will be elaborated later)
 - **TrainValTest.py** is the class abstraction for the train, validation and test process.
 - **Models** folder contains the models you could use, the functions inside this folder should return a keras model.
 - **DataGenerators** folder contains different versions of DataGenerators, for now, we will only use the newest version: **NewDataGenerator**
 - Other files are functional parts wrapped in the main files mentioned above or just test files, you could refer for details if you want.

2. How to use it?  
 * Here is the example script below. (Of course you should add sbatch parameters if you want to run it in sbatch mode)
 ```  
 	python -u /somewhere/RFMLS/train_val_framework/test_framework.py
 	--exp_name train_filtered 
 	--base_path /scratch/RFMLS/dataset100/ 
 	--stats_path /scratch/RFMLS/dataset100/ 
 	--save_path /scratch/somewhere/filtered/ 
 	-d 100 --generator 	new --epochs 10000 -bs 256
   ```
 * So let's get to the parameters  
 		* **exp_name** is the experiment name you can specify, the program will create a **exp_name** folder for you inside **save_path** for saving logs, models etc.  
 		* **base_path** is the path for dictionaries like the partition dictionary for training, validation, testing. You could refer to **/scratch/RFMLS/dataset100/** for further information.  
 		* **stats_path** is the path for dictionaries contains statistics, in the newly generated 100 devices dataset, this is the same as **base_path**.  
 		* **save_path** is the path for saving the model and logs, however, it should be the upper folder of **exp_name**  
 		* **model flag** specifies which model to use, here I only added **'baseline'** and a probably not working version of **'vgg16'** (however I left it here so you could change it easily).  
 			- Here if you want to add your new models, you should modify the function inside **TrainValTest.py**. Line 35, here is a function called **get_model(model\_flag, params={})**, the code should be self self explanatory. You should add a branch in the if-elif blocks. Also, if you need more parameters in **params**, you could add more in Line 89 below in the dictionary:  
 			```
 			new_model = get_model(args.model_flag, {'slice_size':args.slice_size, 
 			'classes':args.devices, 'cnn_stacks':args.cnn_stack,
 			'fc_stacks':args.fc_stack, 
    'channels':args.channels, 'dropout_flag':args.dropout_flag})
 			```  
 			- Meanwhile, you should also add your model in **Models** folder, you must ensure your function returns a keras model.  
 		- **slice_size** is the slice size   
 		- **devices** is the number of devices you want to classify, for the newest dataset is 100 here.   
 		- If you are using **'baseline'** in **model** flag, here are some arguments that you can specify.   
 			- **cnn_stack**, **fc_stack**, **channels**, **dropout_flag** should be self-explanatory.  
 		- For now, **K** and **files\_per\_IO** should just be default value here.  
 		- **cont** specifies you want to continue training an existing model or not.  
 			- If true, you should specify **restore\_model\_from**, which is a keras model in .json format and **restore\_weight\_from** which is the models corresponding weight.  
 		- **generator** specifies which generator to use, for now just use **'new'**  
 		- **preprocessor** specifies what kind of preprocessing you want to add to the slices. If **'no'**, we do not preprocess input slice. If **'tensor'**, we get the II' QQ' IQ' stack of shape (slice_size, slice_size, 3). If **'fft'**, we do fft on the input slice.  
 			- Here what needs to notice is the preprocesser should be related to the input shape of your network!  
 		- **training_strategy**, **'small'** means every epoch we go through all the examples and get one random slice each. So the number of iterations per epoch should be # of examples / batch size. **'big'** means the original method devised by Luca.  
 		- **sampling**, for now we should just leave it as default because the dataset is already balanced.  
 		- **epochs**, **batch_size**, **learning_rate**, **decay** are trivial.  
 		- **multigpu** specifies using multigpu or not.  
 		- **shrink** is the downsampling factor for fast debugging, should leave it 1 for now.  
 		- **test** specifies you want to have a test after training-validation or not.  
   - **flag_error_analysis** specifies you want to analysis testing results or not. 
 
 
 
