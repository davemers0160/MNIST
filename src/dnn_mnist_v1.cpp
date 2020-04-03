#define _CRT_SECURE_NO_WARNINGS
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

// Custom includes
#include "get_platform.h"
#include "file_ops.h"
#include "file_parser.h"
#include "get_current_time.h"
#include "num2string.h"

#include "mnist_net_v0.h"

// dlib includes
#include <dlib/dnn.h>
#include <dlib/data_io.h>

using namespace std;

//----------------------------------------------------------------------------------

std::string version = "06_16_120_84";
//std::string net_name = "mnist_net_" + version;
//std::string net_sync_name = "mnist_sync_" + version;
//std::string logfileName = "mnist_log_" + version + "_";
std::string platform;
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
    std::string sdate, stime;

    std::ofstream DataLogStream;

    const std::string os_file_sep = "/";
    std::string program_root;
    std::string data_directory;     // = "../data";
    std::string save_directory;      // = "../results/";
    std::string net_directory;      // = "../nets/";
    
    const std::vector<int> gpus = { 0 };
    std::vector<uint32_t> filter_num = { 84, 120, 16, 6 };

    typedef std::chrono::duration<double> d_sec;
    auto start_time = chrono::system_clock::now();
    auto stop_time = chrono::system_clock::now();
    auto elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

    // MNIST is broken into two parts, a training set of 60000 images and a test set of
    // 10000 images.  Each image is labeled so that we know what hand written digit is
    // depicted.  These next statements load the dataset into memory.
    std::vector<dlib::matrix<unsigned char>> training_images;
    std::vector<dlib::matrix<unsigned char>> testing_images;
    std::vector<unsigned long> training_labels;
    std::vector<unsigned long> testing_labels;
    
    // check to see if values have been potentially supplied
    // if not then use theb default values
    if (argc == 5)
    {
        filter_num[0] = std::stoi(argv[1]);
        filter_num[1] = std::stoi(argv[2]);
        filter_num[2] = std::stoi(argv[3]);
        filter_num[3] = std::stoi(argv[4]);
        
        version = std::string(argv[4]) + "_" + std::string(argv[3]) + "_" + std::string(argv[2]) + "_" + std::string(argv[1]);
    }

    std::string net_name = "mnist_net_" + version;
    std::string net_sync_name = "mnist_sync_" + version;
    std::string logfileName = "mnist_log_" + version + "_";

    // check the platform
    get_platform(platform);
    uint8_t HPC = 0;

    if (platform.compare(0, 3, "HPC") == 0)
    {
        std::cout << "HPC Platform Detected." << std::endl;
        HPC = 1;
    }

    // setup save variable locations
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
    program_root = get_path(get_path(get_path(std::string(argv[0]), "\\"), "\\"), "\\") + os_file_sep;
    net_directory = program_root + "nets/";
    save_directory = program_root + "results/";
    data_directory = program_root + "data/";
    //data_directory = "../data";

#else    
    if (HPC == 1)
    {
        //HPC version
        program_root = get_path(get_path(get_path(std::string(argv[0]), os_file_sep), os_file_sep), os_file_sep) + os_file_sep;
        data_directory = "../data";
    }
    else
    {
        // Ubuntu
        //program_root = "/home/owner/Projects/MNIST/";     // use this if the get_ubuntu_path does not work
        program_root = get_ubuntu_path();
        data_directory = program_root + "data/";
    }

    net_directory = program_root + "nets/";
    save_directory = program_root + "results/";

#endif

    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "program root:   " << program_root << std::endl;
    std::cout << "data directory: " << data_directory << std::endl;
    std::cout << "net directory:  " << net_directory << std::endl;
    std::cout << "save directory: " << save_directory << std::endl;
    std::cout << std::endl;
    
    try
    {
        // load the data in using the dlib built in function
        dlib::load_mnist_dataset(data_directory, training_images, training_labels, testing_images, testing_labels);

        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "Loaded " << training_images.size() << " training images." << std::endl;
        std::cout << "Loaded " << testing_images.size() << " test images." << std::endl << std::endl;
        
        get_current_time(sdate, stime);
        logfileName = logfileName + sdate + "_" + stime + ".txt";

        std::cout << "Log File: " << (save_directory + logfileName) << std::endl;
        DataLogStream.open((save_directory + logfileName), ios::out | ios::app);

        // Add the date and time to the start of the log file
        DataLogStream << "------------------------------------------------------------------" << std::endl;
        DataLogStream << "Version: 2.0    Date: " << sdate << "    Time: " << stime << std::endl;
        DataLogStream << "------------------------------------------------------------------" << std::endl;
        
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
        
        net_type net;

        config_net(net, filter_num);

        //net_type net(dlib::num_fc_outputs(10),
        //    dlib::num_fc_outputs(filter_num[0]),
        //    dlib::num_fc_outputs(filter_num[1]),
        //    dlib::num_con_outputs(filter_num[2]),
        //    dlib::num_con_outputs(filter_num[3]));

        // And then train it using the MNIST data.  The code below uses mini-batch stochastic
        // gradient descent with an initial learning rate of 0.01 to accomplish this.
        dlib::dnn_trainer<net_type, dlib::sgd> trainer(net, dlib::sgd(), gpus);
        trainer.set_learning_rate(0.01);
        trainer.set_synchronization_file((net_directory + net_sync_name), std::chrono::minutes(5));
        trainer.set_min_learning_rate(0.000001);
        trainer.set_mini_batch_size(512 * gpus.size());
        trainer.set_max_num_epochs(100);
        trainer.set_iterations_without_progress_threshold(2000);

        trainer.be_verbose();
        
        std::cout << std::endl << "------------------------------------------------------------------" << std::endl;
        std::cout << trainer << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;
        DataLogStream << trainer << std::endl;
        DataLogStream << "------------------------------------------------------------------" << std::endl;

        std::cout << "Net Name: " << net_name << std::endl;
        std::cout << net << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;
        DataLogStream << "Net Name: " << net_name << std::endl;
        DataLogStream << net << std::endl;
        DataLogStream << "------------------------------------------------------------------" << std::endl;

        //init_gorgon((save_directory + "gorgon_mnist_"));

        // Finally, this line begins training.  By default, it runs SGD with our specified
        // learning rate until the loss stops decreasing.  Then it reduces the learning rate by
        // a factor of 10 and continues running until the loss stops decreasing again.  It will
        // keep doing this until the learning rate has dropped below the min learning rate
        // defined above or the maximum number of epochs as been executed (defaulted to 10000). 
        if (true)
        {
            std::cout << "Starting Training..." << std::endl;
            start_time = chrono::system_clock::now();
            trainer.train(training_images, training_labels);
            stop_time = chrono::system_clock::now();
        
            //std::vector<float> g1_weights = gc_01.get_params(net);
            //std::vector<float> g2_weights = gc_02.get_params(net);
            //std::vector<float> g3_weights = gc_03.get_params(net);
            //std::vector<float> g4_weights = gc_04.get_params(net);
            //std::vector<float> g5_weights = gc_05.get_params(net);
            //save_gorgon(net, 4);

            //trainer.set_max_num_epochs(6);
            //trainer.train(training_images, training_labels);
            //stop_time = chrono::system_clock::now();

            //save_gorgon(net, 8);

            //std::vector<float> g1_weights2 = gc_01.get_params(net);
            //std::vector<float> g2_weights2 = gc_02.get_params(net);
            //std::vector<float> g3_weights2 = gc_03.get_params(net);
            //std::vector<float> g4_weights2 = gc_04.get_params(net);
            //std::vector<float> g5_weights2 = gc_05.get_params(net);

            // At this point our net object should have learned how to classify MNIST images.  But
            // before we try it out let's save it to disk.  Note that, since the trainer has been
            // running images through the network, net will have a bunch of state in it related to
            // the last batch of images it processed (e.g. outputs from each layer).  Since we
            // don't care about saving that kind of stuff to disk we can tell the network to forget
            // about that kind of transient data so that our file will be smaller.  We do this by
            // "cleaning" the network before saving it.

            // try copying a previous set of filters into the network
            //auto& layer_details = dlib::layer<1>(net).layer_details();
            //auto& layer_params = layer_details.get_layer_params();
            //float *params_data = layer_params.host();
            //dlib::alias_tensor_instance tmp = (dlib::alias_tensor_instance)layer_details.get_layer_params();

            //float *params_data = dlib::mat(layer_params.host());
            //float *params_data = dlib::mat(layer_details.get_layer_params());

            

            //for (uint64_t jdx = 0; jdx < g1_weights.size(); ++jdx)
            //{
            //    params_data[jdx] = g1_weights[jdx];
            //}


            net.clean();

            dlib::serialize((net_directory + net_name + ".dat")) << net;
        }

        //close_gorgon();

        // Now if we later wanted to recall the network from disk we can simply say:
        // deserialize("mnist_network.dat") >> net;

        elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);
        std::cout << "Training Complete.  Elapsed Time: " << elapsed_time.count() / 3600 << " hours" << std::endl;
        DataLogStream << "Training Complete.  Elapsed Time: " << elapsed_time.count() / 3600 << " hours" << std::endl;

        std::cout << "------------------------------------------------------------------" << std::endl;

        net_type test_net;

        std::string test_net_name = (net_directory + net_name + ".dat"); // "D:/Projects/MNIST/nets/mnist_net_04_13_76_56.dat";
        dlib::deserialize(test_net_name) >> test_net;

        std::cout << test_net_name << std::endl;
        std::cout << test_net << std::endl;

        DataLogStream << "------------------------------------------------------------------" << std::endl;
        DataLogStream << "test_net: " << test_net_name << std::endl;
        DataLogStream << test_net << std::endl;

        // Now let's run the training images through the network.  This statement runs all the
        // images through it and asks the loss layer to convert the network's raw output into
        // labels.
        std::cout << std::endl << "Analyzing Training Results..." << std::endl << std::endl;
        // run the first set through to prime the GPU
        dlib::matrix<double, 1, 3> training_results = eval_net_performance(test_net, training_images, training_labels);

        uint16_t time_num = 50;
        double avg_train_time = 0.0;
        for (uint32_t idx = 0; idx < time_num; ++idx)
        {
            start_time = chrono::system_clock::now();
            test_net(training_images);
            stop_time = chrono::system_clock::now();
            elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);
            avg_train_time += elapsed_time.count();
            std::cout << ".";
        }
        avg_train_time = avg_train_time / (double)time_num;
        std::cout << endl;

        //double avg_train_time = elapsed_time.count() / (double)training_images.size();

        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "Average run time:   " << avg_train_time << std::endl;
        std::cout << "Training num_right: " << training_results(0,0) << std::endl;
        std::cout << "Training num_wrong: " << training_results(0,1) << std::endl;
        std::cout << "Training accuracy:  " << training_results(0,2) << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;

        double avg_test_time = 0.0;

        // Let's also see if the network can correctly classify the testing images.  Since
        // MNIST is an easy dataset, we should see at least 99% accuracy.
        std::cout << std::endl << "Analyzing Test Results..." << std::endl << std::endl;
        dlib::matrix<double, 1, 3> test_results = eval_net_performance(test_net, testing_images, testing_labels);

        for (uint32_t idx = 0; idx < time_num; ++idx)
        {
            start_time = chrono::system_clock::now();
            test_net(testing_images);
            stop_time = chrono::system_clock::now();
            elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);
            avg_test_time += elapsed_time.count();
            std::cout << ".";
        }
        avg_test_time = avg_test_time / (double)time_num;
        std::cout << endl;
        //double avg_test = elapsed_time.count() / (double)testing_images.size();

        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "Average run time:   " << avg_test_time << std::endl;
        std::cout << "Test num_right: " << test_results(0,0) << std::endl;
        std::cout << "Test num_wrong: " << test_results(0,1) << std::endl;
        std::cout << "Test accuracy:  " << test_results(0,2) << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;

        DataLogStream << "------------------------------------------------------------------" << std::endl;
        DataLogStream << training_results(0, 0) << ", " << training_results(0, 1) << ", " << training_results(0, 2) << ", "
                      << test_results(0, 0) << ", " << test_results(0, 1) << ", " << test_results(0, 2) << ", "
                      << avg_train_time << ", " << avg_test_time << std::endl;

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

