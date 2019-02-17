#ifndef NET_DEFINITION_H
#define NET_DEFINITION_H

#include <cstdint>
#include <string>

#include "dlib/matrix.h"
#include "dlib/dnn.h"
#include "dlib/dnn/core.h"

using net_type = dlib::loss_multiclass_log<
    dlib::fc<10,
    dlib::htan<dlib::fc<84,
    dlib::sig<dlib::con<120, 5, 5, 1, 1,
    dlib::sig<dlib::max_pool<2, 2, 2, 2, dlib::con<16, 5, 5, 1, 1,
    dlib::sig<dlib::max_pool<2, 2, 2, 2, dlib::con<6, 5, 5, 1, 1,
    dlib::input<dlib::matrix<unsigned char>>
    >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

template <typename net_type>
void config_net(net_type &net, std::vector<uint32_t> params)
{
    net = net_type(dlib::num_fc_outputs(10),
        dlib::num_fc_outputs(params[0]),
        dlib::num_con_outputs(params[1]),
        dlib::num_con_outputs(params[2]),
        dlib::num_con_outputs(params[3])
    );

}   // end of config_net

// ----------------------------------------------------------------------------------------

#endif
