#ifndef REACTORCVODE_H
#define REACTORCVODE_H

#include "AMReX_Sundials.H"
#include "ReactorBase.H"
#include "ReactorCvodeUtils.H"
#include "ReactorCvodePreconditioner.H"
#include "ReactorCvodeJacobian.H"
#include "ReactorCvodeCustomLinSolver.H"
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunmatrix/sunmatrix_sparse.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunlinsol/sunlinsol_spbcgs.h>
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h>

#ifdef AMREX_USE_HIP
#include <nvector/nvector_hip.h>
#endif

#ifdef AMREX_USE_SYCL
#include <nvector/nvector_sycl.h>
#endif

#ifdef PELE_USE_KLU
#include "klu.h"
#include <sunlinsol/sunlinsol_klu.h>
#endif

#ifdef PELE_USE_MAGMA
#include <sunmatrix/sunmatrix_magmadense.h>
#include <sunlinsol/sunlinsol_magmadense.h>
#endif

#ifdef PELE_USE_GINKGO
#include <sunmatrix/sunmatrix_ginkgoblock.hpp>
#include <sunlinsol/sunlinsol_ginkgoblock.hpp>
#endif

#ifdef AMREX_USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <cuda_runtime_api.h>
#include <nvector/nvector_cuda.h>
#include <sunmatrix/sunmatrix_cusparse.h>
#include <sunlinsol/sunlinsol_cusolversp_batchqr.h>
#endif

namespace pele::physics::reactions {

class ReactorCvode : public ReactorBase::Register<ReactorCvode>
{
public:
  int cvode_call_index = 0;
  static std::string identifier() { return "ReactorCvode"; }

  using Ordering = utils::YCOrder;

  int init(int reactor_type, int ncells) override;

  void close() override;

  void print_final_stats(void* cvodemem, bool print_ls_stats);
  void print_final_stats(void* cvodemem) override
  {
    print_final_stats(cvodemem, true);
  }

  int react(
    const amrex::Box& box,
    amrex::Array4<amrex::Real> const& rY_in,
    amrex::Array4<amrex::Real> const& rYsrc_in,
    amrex::Array4<amrex::Real> const& T_in,
    amrex::Array4<amrex::Real> const& rEner_in,
    amrex::Array4<amrex::Real> const& rEner_src_in,
    amrex::Array4<amrex::Real> const& FC_in,
    amrex::Array4<int> const& mask,
    amrex::Real& dt_react,
    amrex::Real& time
#ifdef AMREX_USE_GPU
    ,
    amrex::gpuStream_t stream
#endif
    ) override;

  int react(
    amrex::Real* rY_in,
    amrex::Real* rYsrc_in,
    amrex::Real* rX_in,
    amrex::Real* rX_src_in,
    amrex::Real& dt_react,
    amrex::Real& time,
    int ncells
#ifdef AMREX_USE_GPU
    ,
    amrex::gpuStream_t stream
#endif
    ) override;

  static int
  cF_RHS(amrex::Real t, N_Vector y_in, N_Vector ydot, void* user_data);

  void flatten(
    const amrex::Box& box,
    const int ncells,
    amrex::Array4<const amrex::Real> const& rhoY,
    amrex::Array4<const amrex::Real> const& frcExt,
    amrex::Array4<const amrex::Real> const& temperature,
    amrex::Array4<const amrex::Real> const& rhoE,
    amrex::Array4<const amrex::Real> const& frcEExt,
    amrex::Real* y_vect,
    amrex::Real* src_vect,
    amrex::Real* vect_energy,
    amrex::Real* src_vect_energy) override
  {
    flatten_ops.flatten(
      box, ncells, m_reactor_type, m_clean_init_massfrac, rhoY, frcExt,
      temperature, rhoE, frcEExt, y_vect, src_vect, vect_energy,
      src_vect_energy);
  }

  void unflatten(
    const amrex::Box& box,
    const int ncells,
    amrex::Array4<amrex::Real> const& rhoY,
    amrex::Array4<amrex::Real> const& temperature,
    amrex::Array4<amrex::Real> const& rhoE,
    amrex::Array4<amrex::Real> const& frcEExt,
    amrex::Array4<amrex::Real> const& FC_in,
    amrex::Real* y_vect,
    amrex::Real* vect_energy,
    long int* FCunt,
    amrex::Real dt) override
  {
    flatten_ops.unflatten(
      box, ncells, m_reactor_type, m_clean_init_massfrac, rhoY, temperature,
      rhoE, frcEExt, FC_in, y_vect, vect_energy, FCunt, dt);
  }

private:
  void checkCvodeSolveType();
  void printExtraSolveInformation() const;
  int setCvodeOptions(void* cvode_mem, bool analytical_jacobian) const;
  void allocUserData(
    CVODEUserData* udata,
    int a_ncells
#ifdef AMREX_USE_GPU
    ,
    SUNMatrix& a_A,
    amrex::gpuStream_t stream
#endif
  ) const;
  void freeUserData(CVODEUserData* data_wk);

  int m_reactor_type{0};
  utils::FlattenOps<Ordering> flatten_ops;
  bool m_clean_init_massfrac{false};
  int solve_type{-1};
  int analytical_jacobian{0};
  int precond_type{-1};

  // CVODE Integrator tolerances
  amrex::Real relTol{1e-6};
  amrex::Real absTol{1e-10};

#ifdef PELE_USE_GINKGO
  std::shared_ptr<const gko::Executor> gko_exec;
#endif

#ifndef AMREX_USE_GPU
  N_Vector y{nullptr};
  SUNLinearSolver LS{nullptr};
  SUNNonlinearSolver NLS{nullptr};
  SUNMatrix A{nullptr};
  void* cvode_mem{nullptr};
  CVODEUserData* udata_g{nullptr};
#endif

  int max_order{2};
  int max_err_test_fails{100};
  int max_num_steps{100000};
  bool sunprofiler_print_every_step{false};
  bool linear_solver_scaling{true}; // temporary, remove after debugging

  // This block of CVODE parameters are initialized to the CVODE default values.
  // WARNING: it is unwise to change these without a really good reason. The
  // default values have been empircally proven to be robust and optimal for
  // many types of problems. It may be possible to achieve better performance
  // for a specific problem with different values, so the parameters can be
  // changed, but do so with extreme care.

  // CVODE integrator parameters related to time step size selection:
  // https://sundials.readthedocs.io/en/latest/cvode/Usage/index.html#id10
  amrex::Real eta_cf{0.25};
  amrex::Real eta_max_fx{1.5};
  amrex::Real eta_max_gs{10.0};
  amrex::Real eta_min_ef{0.1};
  amrex::Real eta_min_fx{0.0};
  amrex::Real eta_min{1.0};
  // CVODE integrator parameters controlling linear solver
  // https://sundials.readthedocs.io/en/latest/cvode/Usage/index.html#id7
  amrex::Real dgmax{
    0.3}; // Max allowed gamma change without a linear solver setup
  amrex::Real epslin{0.0}; // Ratio between linear and nonlinear tolerances
  int msbj{51};            // Jacobian / preconditioner update frequency
  int msbp{
    20};       // Linear solver setup frequency (e.g. refactorization frequency)
  int maxl{5}; // max # of Krylov subspace vectors (for GMRES)
  // CVODE integrator parameters controlling nonlinear solver
  // https://sundials.readthedocs.io/en/latest/cvode/Usage/index.html#id9
  amrex::Real nonlin_conv_coef{
    0.1}; // Coefficient in the nonlinear convergence test
  int max_nls_iters{
    3}; // max # of nonlinear solver iterations before considering
  int max_conv_fails{
    10};               // max # of nonlinear solver convergence failures before
  int max_fp_accel{0}; // max # of Anderson Acceleration vectors (if using the
                       // fixed-point nonlinear solver)
};
} // namespace pele::physics::reactions
#endif
