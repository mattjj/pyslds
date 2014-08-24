#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "nptypes.h"
#include "util.h"

using namespace Eigen;
using namespace nptypes;

// TODO compare LDLT and LLT speed
// TODO implement a condition_on method
// TODO figure out how to Map so we can avoid alloc

template <typename Type>
class dummy
{
    public:

    static void kf_resample_lds(
            int T, int D, int P,
            Type *As, Type *BBTs, Type *Cs, Type *DDTs,
            Type *data, Type *randseq, Type *out)
    {
        NPMatrix<Type> edata(data,T,D);
        NPMatrix<Type> eout(out,T,P);
        NPMatrix<Type> erandseq(randseq,T,P);

        // TODO
    }
}
