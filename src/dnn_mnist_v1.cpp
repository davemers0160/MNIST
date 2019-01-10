// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the
    dlib C++ Library.  In it, we will train the venerable LeNet convolutional
    neural network to recognize hand written digits.  The network will take as
    input a small image and classify it as one of the 10 numeric digits between
    0 and 9.

    The specific network we will run is from the paper
        LeCun, Yann, et al. "Gradient-based learning applied to document recognition."
        Proceedings of the IEEE 86.11 (1998): 2278-2324.
    except that we replace the sigmoid non-linearities with rectified linear units. 

    These tools will use CUDA and cuDNN to drastically accelerate network
    training and testing.  CMake should automatically find them if they are
    installed and configure things appropriately.  If not, the program will
    still run but will be much slower to execute.
*/

#include <cstdint>
#include <iostream>
#include <string>

#include <dlib/dnn.h>
#include <dlib/data_io.h>

using namespace std;

//----------------------------------------------------------------------------------

template <typename net_type>
dlib::matrix<double, 1, 3> eval_net_performance(net_type &net, std::vector<dlib::matrix<unsigned char>> input_images, std::vector<unsigned long> input_labels)
{
    std::vector<unsigned long> predicted_labels = net(input_images);
    int num_right = 0;
    int num_wrong = 0;
    // And then let's see if it classified them correctly.
    for (size_t i = 0; i < input_images.size(); ++i)
    {
        if (predicted_labels[i] == input_labels[i])
            ++num_right;
        else
            ++num_wrong;
        
    }
    // std::cout << "training num_right: " << num_right << std::endl;
    // std::cout << "training num_wrong: " << num_wrong << std::endl;
    // std::cout << "training accuracy:  " << num_right/(double)(num_right+num_wrong) << std::endl;
    
    dlib::matrix<double, 1, 3> results;
    results = (double)num_right, (double)num_wrong, (double)num_right/(double)(num_right+num_wrong);
    
    return results;
    
}   // end of eval_net_performance

//----------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    // This example is going to run on the MNIST dataset.  
    // if (argc != 2)
    // {
        // cout << "This example needs the MNIST dataset to run!" << endl;
        // cout << "You can get MNIST from http://yann.lecun.com/exdb/mnist/" << endl;
        // cout << "Download the 4 files that comprise the dataset, decompress them, and" << endl;
        // cout << "put them in a folder.  Then give that folder as input to this program." << endl;
        // return 1;
    // }

    std::string data_directory = "../data";
    std::string save_directry = "../results/";
    std::string net_directory = "../nets/";
    
    const std::vector<int> gpus = { 0 };
    
    // MNIST is broken into two parts, a training set of 60000 images and a test set of
    // 10000 images.  Each image is labeled so that we know what hand written digit is
    // depicted.  These next statements load the dataset into memory.
    std::vector<dlib::matrix<unsigned char>> training_images;
    std::vector<dlib::matrix<unsigned char>> testing_images;
    std::vector<unsigned long> training_labels;
    std::vector<unsigned long> testing_labels;
    
    // load the data in using the dlib built in function
    dlib::load_mnist_dataset(data_directory, training_images, training_labels, testing_images, testing_labels);


    // Now let's define the LeNet.  Broadly speaking, there are 3 parts to a network
    // definition.  The loss layer, a bunch of computational layers, and then an input
    // layer.  You can see these components in the network definition below.  
    // 
    // The input layer here says the network expects to be given matrix<unsigned char>
    // objects as input.  In general, you can use any dlib image or matrix type here, or
    // even define your own types by creating custom input layers.
    //
    // Then the middle layers define the computation the network will do to transform the
    // input into whatever we want.  Here we run the image through multiple convolutions,
    // ReLU units, max pooling operations, and then finally a fully connected layer that
    // converts the whole thing into just 10 numbers.  
    // 
    // Finally, the loss layer defines the relationship between the network outputs, our 10
    // numbers, and the labels in our dataset.  Since we selected loss_multiclass_log it
    // means we want to do multiclass classification with our network.   Moreover, the
    // number of network outputs (i.e. 10) is the number of possible labels.  Whichever
    // network output is largest is the predicted label.  So for example, if the first
    // network output is largest then the predicted digit is 0, if the last network output
    // is largest then the predicted digit is 9.  
    using net_type = dlib::loss_multiclass_log<
                                dlib::fc<10,        
                                dlib::prelu<dlib::fc<84,   
                                dlib::prelu<dlib::fc<120,  
                                dlib::max_pool<2,2,2,2,dlib::prelu<dlib::con<16,5,5,1,1,
                                dlib::max_pool<2,2,2,2,dlib::prelu<dlib::con<6,5,5,1,1,
                                dlib::input<dlib::matrix<unsigned char>> 
                                >>>>>>>>>>>>;

    try
    {
        // So with that out of the way, we can make a network instance.
        net_type net;
        // And then train it using the MNIST data.  The code below uses mini-batch stochastic
        // gradient descent with an initial learning rate of 0.01 to accomplish this.
        dlib::dnn_trainer<net_type, dlib::sgd> trainer(net, dlib::sgd(), gpus);
        trainer.set_learning_rate(0.0001);
        trainer.set_min_learning_rate(0.000001);
        trainer.set_mini_batch_size(4096 * gpus.size());
        trainer.set_max_num_epochs(10000);
        trainer.set_iterations_without_progress_threshold(2000);
        trainer.set_synchronization_file((net_directory + "mnist_sync"), std::chrono::minutes(2));
        trainer.be_verbose();
        
        std::cout << std::endl << trainer << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;

        std::cout << net << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;

        // Finally, this line begins training.  By default, it runs SGD with our specified
        // learning rate until the loss stops decreasing.  Then it reduces the learning rate by
        // a factor of 10 and continues running until the loss stops decreasing again.  It will
        // keep doing this until the learning rate has dropped below the min learning rate
        // defined above or the maximum number of epochs as been executed (defaulted to 10000). 
        trainer.train(training_images, training_labels);

        // At this point our net object should have learned how to classify MNIST images.  But
        // before we try it out let's save it to disk.  Note that, since the trainer has been
        // running images through the network, net will have a bunch of state in it related to
        // the last batch of images it processed (e.g. outputs from each layer).  Since we
        // don't care about saving that kind of stuff to disk we can tell the network to forget
        // about that kind of transient data so that our file will be smaller.  We do this by
        // "cleaning" the network before saving it.
        net.clean();
        dlib::serialize((net_directory+"mnist_network.dat")) << net;

        // Now if we later wanted to recall the network from disk we can simply say:
        // deserialize("mnist_network.dat") >> net;


        // Now let's run the training images through the network.  This statement runs all the
        // images through it and asks the loss layer to convert the network's raw output into
        // labels.
        std::cout << std::endl << "Analyzing Training Results..." << std::endl << std::endl;

        dlib::matrix<double, 1, 3> training_results = eval_net_performance(net, training_images, training_labels);
        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "Training num_right: " << training_results(0,0) << std::endl;
        std::cout << "Training num_wrong: " << training_results(0,1) << std::endl;
        std::cout << "Training accuracy:  " << training_results(0,2) << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;

        // Let's also see if the network can correctly classify the testing images.  Since
        // MNIST is an easy dataset, we should see at least 99% accuracy.
        std::cout << std::endl << "Analyzing Test Results..." << std::endl << std::endl;

        dlib::matrix<double, 1, 3> test_results = eval_net_performance(net, testing_images, testing_labels);

        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "Test num_right: " << test_results(0,0) << std::endl;
        std::cout << "Test num_wrong: " << test_results(0,1) << std::endl;
        std::cout << "Test accuracy:  " << test_results(0,2) << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;


        // Finally, you can also save network parameters to XML files if you want to do
        // something with the network in another tool.  For example, you could use dlib's
        // tools/convert_dlib_nets_to_caffe to convert the network to a caffe model.
        //net_to_xml(net, "lenet.xml");
        
    }
    catch(std::exception& e)
    {
        std::cout << std::endl << e.what() << std::endl;
        std::cin.ignore();
    }

    std::cout << std::endl << "Program complete.  Press Enter to close." << std::endl;
    std::cin.ignore();
    return 0;

}   // end of main

