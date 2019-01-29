#ifndef NET_DEFINITION_H
#define NET_DEFINITION_H

#include <cstdint>
#include <string>

#include "dlib/matrix.h"
#include "dlib/dnn.h"
#include "dlib/dnn/core.h"

using net_type = dlib::loss_multiclass_log<
    dlib::fc<10,
    dlib::prelu<dlib::fc<84,
    dlib::prelu<dlib::fc<120,
    dlib::max_pool<2, 2, 2, 2, dlib::prelu<dlib::con<16, 5, 5, 1, 1,
    dlib::max_pool<2, 2, 2, 2, dlib::prelu<dlib::con<6, 5, 5, 1, 1,
    dlib::input<dlib::matrix<unsigned char>>
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

#endif
