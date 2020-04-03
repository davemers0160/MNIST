#ifndef NET_DEFINITION_H
#define NET_DEFINITION_H

#include <cstdint>
#include <string>

#include "gorgon_capture.h"

#include "dlib/matrix.h"
#include "dlib/dnn.h"
#include "dlib/dnn/core.h"

using net_type = dlib::loss_multiclass_log<
    dlib::fc<10,
    dlib::prelu<dlib::fc<84,
    dlib::prelu<dlib::fc<120,
    dlib::max_pool<2, 2, 2, 2, dlib::prelu<dlib::con<16, 5, 5, 1, 1,
    dlib::max_pool<2, 2, 2, 2, dlib::prelu<dlib::con<6, 5, 5, 1, 1,
    dlib::tag1<dlib::input<dlib::matrix<unsigned char>>>
    >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

template <typename net_type>
void config_net(net_type &net, std::vector<uint32_t> params)
{
    net = net_type(dlib::num_fc_outputs(10),
        dlib::num_fc_outputs(params[0]),
        dlib::num_fc_outputs(params[1]),
        dlib::num_con_outputs(params[2]),
        dlib::num_con_outputs(params[3])
    );

}   // end of config_net

// ----------------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------------
//  GORGON Functions
// ----------------------------------------------------------------------------------------

// N, K, NR, NC
gorgon_capture<1> gc_01(84,10);
gorgon_capture<3> gc_02(120,84);
gorgon_capture<5> gc_03(784,120);
gorgon_capture<8> gc_04(16,6,5,5);
gorgon_capture<11> gc_05(6,1,5,5);


void init_gorgon(std::string save_location)
{
    gc_01.init((save_location + "l01"));
    gc_02.init((save_location + "l03"));
    gc_03.init((save_location + "l05"));
    gc_04.init((save_location + "l08"));
    gc_05.init((save_location + "l11"));

}

template<typename net_type>
void save_gorgon(net_type &net, uint64_t one_step_calls)
{
    gc_01.save_params(net, one_step_calls);
    gc_02.save_params(net, one_step_calls);
    gc_03.save_params(net, one_step_calls);
    gc_04.save_params(net, one_step_calls);
    gc_05.save_params(net, one_step_calls);

}


void close_gorgon(void)
{
    gc_01.close_stream();
    gc_02.close_stream();
    gc_03.close_stream();
    gc_04.close_stream();
    gc_05.close_stream();

}




#endif
