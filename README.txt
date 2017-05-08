# Step1: get the data
caffe/data/mnist/get_mnist.sh

# Step2: The data would be stored into 
# launch the shell script
create_mnist.sh 
caffe/examples/mnist/mnist_test_lmdb
caffe/examples/mnist/mnist_train_lmdb

# Step3: create the model and solver prototxt files

# Step4: draw picture for model
python drawnet.py model_prototxt_file pic01.png --rankdir=LR

# Step5: evaluate the time 
caffe time -model model_file -iterations 100

# Step6: train the model
caffe train solver_file 2>&1 | tee train.log

# Step7: draw statistical chart
$ python ../caffe/tools/extra/plot_training_log.py.example 0 accuracy_iters.png train.log
