#include <Eigen/Dense>

#include "nptypes.h"

using namespace Eigen;
using namespace nptypes;

// TODO compare LDLT and LLT speed
// TODO figure out how to Map so we can avoid alloc

template <typename Type>
class dummy
{
    public:

    static void condition_on(
            int D, int P,
            Type *mu_x, Type *sigma_x, Type *A, Type *sigma_obs, Type *y,
            Type *mu_out, Type *sigma_out)
    {
        // TODO should be using symmetric matrix types!
        // TODO check allocation, which probably happens several times
        // TODO don't redo the llt
        NPVector<Type> emu_x(mu_x,P);
        NPMatrix<Type> esigma_x(sigma_x,P,P);
        NPMatrix<Type> eA(A,D,P);
        NPMatrix<Type> esigma_obs(sigma_obs,D,D);
        NPVector<Type> ey(y,D);
        NPVector<Type> emu_out(mu_out,P);
        NPMatrix<Type> esigma_out(sigma_out,P,P);

        Type sigma_xy[D*P] __attribute__((aligned(32)));
        NPMatrix<Type> esigma_xy(sigma_xy,P,D);
        Type sigma_yy[D*D] __attribute__((aligned(32)));
        NPMatrix<Type> esigma_yy(sigma_yy,D,D);

        esigma_xy.noalias() = esigma_x * eA.transpose();
        esigma_yy.noalias() = eA * esigma_x * eA.transpose() + esigma_obs;
        emu_out = emu_x + esigma_xy * esigma_yy.llt().solve(ey - eA * emu_x);
        esigma_out = esigma_x - esigma_xy * esigma_yy.llt().solve(esigma_xy.transpose());
        esigma_out.template triangularView<Lower>() = esigma_out.transpose();
    }

//    static void kf_resample_lds(
//            int T, int D, int P,
//            Type *As, Type *BBTs, Type *Cs, Type *DDTs,
//            Type *data, Type *randseq, Type *out)
//    {
//        NPMatrix<Type> edata(data,T,D);
//        NPMatrix<Type> eout(out,T,P);
//        NPMatrix<Type> erandseq(randseq,T,P);
//
//        // TODO
//    }
};
