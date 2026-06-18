// Standalone prototype: mixed-precision multigrid-preconditioned CG on a
// Poisson matrix, to nail the Ginkgo API + measure VRAM before porting to OGL.
// Modes: double | dpsp (double finest, float coarse) | single (all float).
// Measures device free-memory delta (Intel SYCL ext) = real VRAM used.
#include <ginkgo/ginkgo.hpp>
#include <sycl/sycl.hpp>
#include <cstdio>
#include <string>

using vt=double; using it=int;
using mtxd=gko::matrix::Csr<double,it>;
using cg=gko::solver::Cg<double>;
using mg=gko::solver::Multigrid;
using pgmd=gko::multigrid::Pgm<double,it>;
using pgmf=gko::multigrid::Pgm<float,it>;
using ird=gko::solver::Ir<double>;
using irf=gko::solver::Ir<float>;
using bjd=gko::preconditioner::Jacobi<double,it>;
using bjf=gko::preconditioner::Jacobi<float,it>;
using cgf=gko::solver::Cg<float>;
using Iter=gko::stop::Iteration;

static size_t free_mib(sycl::device& d){
  try { return d.get_info<sycl::ext::intel::info::device::free_memory>()/(1024*1024); }
  catch(...) { return 0; }
}

int main(int argc,char**argv){
  std::string mode = argc>1?argv[1]:"double";
  int N = argc>2?std::atoi(argv[2]):2000;     // N*N rows
  int maxit = argc>3?std::atoi(argv[3]):100;
  auto omp = gko::OmpExecutor::create();
  auto exec = gko::DpcppExecutor::create(0, omp);
  auto dev = exec->get_queue()->get_device();
  size_t f0 = free_mib(dev);

  // Poisson 5-point
  int n=N*N;
  gko::matrix_data<double,it> md(gko::dim<2>{(gko::dim<2>::dimension_type)n,(gko::dim<2>::dimension_type)n});
  md.nonzeros.reserve(5*n);
  for(int j=0;j<N;++j)for(int i=0;i<N;++i){int r=j*N+i;
    md.nonzeros.push_back({r,r,4.0});
    if(i>0)md.nonzeros.push_back({r,r-1,-1.0}); if(i<N-1)md.nonzeros.push_back({r,r+1,-1.0});
    if(j>0)md.nonzeros.push_back({r,r-N,-1.0}); if(j<N-1)md.nonzeros.push_back({r,r+N,-1.0});}
  auto A=gko::share(mtxd::create(exec)); A->read(md); exec->synchronize();
  auto b=gko::share(gko::matrix::Dense<double>::create(exec,gko::dim<2>{(gko::dim<2>::dimension_type)n,1})); b->fill(1.0);
  auto x=gko::share(gko::matrix::Dense<double>::create(exec,gko::dim<2>{(gko::dim<2>::dimension_type)n,1})); x->fill(0.0);
  size_t f1=free_mib(dev);

  // FULL FLOAT: convert matrix+vectors to float, solve entirely in float.
  if(mode=="fullfloat"){
    using mtxf=gko::matrix::Csr<float,it>; using df=gko::matrix::Dense<float>;
    auto Af=gko::share(mtxf::create(exec)); Af->copy_from(A);
    auto bf=gko::share(df::create(exec,gko::dim<2>{(gko::dim<2>::dimension_type)n,1})); bf->fill(1.0f);
    auto xf=gko::share(df::create(exec,gko::dim<2>{(gko::dim<2>::dimension_type)n,1})); xf->fill(0.0f);
    A.reset(); b.reset(); x.reset(); exec->synchronize();   // free the double copies
    size_t ff1=free_mib(dev);
    auto smf=gko::share(irf::build().with_solver(bjf::build().with_max_block_size(1u).on(exec)).with_relaxation_factor(0.9f).with_criteria(Iter::build().with_max_iters(1u)).on(exec));
    auto cof=gko::share(cgf::build().with_preconditioner(bjf::build().with_max_block_size(1u).on(exec)).with_criteria(Iter::build().with_max_iters(20u)).on(exec));
    auto mgf=gko::share(mg::build().with_max_levels(20u).with_min_coarse_rows(64000u)
      .with_pre_smoother(smf).with_post_uses_pre(true)
      .with_mg_level(pgmf::build().with_deterministic(false).on(exec))
      .with_coarsest_solver(cof).with_criteria(Iter::build().with_max_iters(1u)).on(exec));
    auto sf=cgf::build().with_criteria(Iter::build().with_max_iters((unsigned)maxit),
        gko::stop::ResidualNorm<float>::build().with_reduction_factor(1e-6f))
        .with_preconditioner(mgf).on(exec)->generate(Af);
    size_t ff2=free_mib(dev);
    auto lg=gko::share(gko::log::Convergence<float>::create()); sf->add_logger(lg);
    sf->apply(bf,xf); exec->synchronize();
    size_t ff3=free_mib(dev);
    printf("mode=fullfloat N=%d rows=%d | iters=%ld | VRAM: mat+vec=%zuMiB precond_gen=%zuMiB total_used=%zuMiB (free %zu->%zu)\n",
      N,n,(long)lg->get_num_iterations(), f0-ff1, ff1-ff2, f0-ff3, f0, ff3);
    return 0;
  }

  std::shared_ptr<gko::LinOpFactory> mgfac;
  auto crit=[&](unsigned k){return gko::share(Iter::build().with_max_iters(k).on(exec));};

  if(mode=="double"){
    auto sm=gko::share(ird::build().with_solver(bjd::build().with_max_block_size(1u).on(exec)).with_relaxation_factor(0.9).with_criteria(Iter::build().with_max_iters(1u)).on(exec));
    auto co=gko::share(cg::build().with_preconditioner(bjd::build().with_max_block_size(1u).on(exec)).with_criteria(Iter::build().with_max_iters(20u)).on(exec));
    mgfac=mg::build().with_max_levels(20u).with_min_coarse_rows(64000u)
      .with_pre_smoother(sm).with_post_uses_pre(true)
      .with_mg_level(pgmd::build().with_deterministic(false).on(exec))
      .with_coarsest_solver(co).with_criteria(Iter::build().with_max_iters(1u)).on(exec);
  } else if(mode=="dpsp"){
    // level 0 double, levels 1+ float
    auto smd=gko::share(ird::build().with_solver(bjd::build().with_max_block_size(1u).on(exec)).with_relaxation_factor(0.9).with_criteria(Iter::build().with_max_iters(1u)).on(exec));
    auto smf=gko::share(irf::build().with_solver(bjf::build().with_max_block_size(1u).on(exec)).with_relaxation_factor(0.9f).with_criteria(Iter::build().with_max_iters(1u)).on(exec));
    auto cof=gko::share(cgf::build().with_preconditioner(bjf::build().with_max_block_size(1u).on(exec)).with_criteria(Iter::build().with_max_iters(20u)).on(exec));
    mgfac=mg::build().with_max_levels(20u).with_min_coarse_rows(64000u)
      .with_pre_smoother(smd,smf).with_post_uses_pre(true)
      .with_mg_level(pgmd::build().with_deterministic(false).on(exec),
                     pgmf::build().with_deterministic(false).on(exec))
      .with_level_selector([](const gko::size_type lvl,const gko::LinOp*){return lvl==0?0:1;})
      .with_coarsest_solver(cof).with_criteria(Iter::build().with_max_iters(1u)).on(exec);
  } else { // single = all float
    auto smf=gko::share(irf::build().with_solver(bjf::build().with_max_block_size(1u).on(exec)).with_relaxation_factor(0.9f).with_criteria(Iter::build().with_max_iters(1u)).on(exec));
    auto cof=gko::share(cgf::build().with_preconditioner(bjf::build().with_max_block_size(1u).on(exec)).with_criteria(Iter::build().with_max_iters(20u)).on(exec));
    mgfac=mg::build().with_max_levels(20u).with_min_coarse_rows(64000u)
      .with_pre_smoother(smf).with_post_uses_pre(true)
      .with_mg_level(pgmf::build().with_deterministic(false).on(exec))
      .with_coarsest_solver(cof).with_criteria(Iter::build().with_max_iters(1u)).on(exec);
  }

  auto solver=cg::build().with_criteria(Iter::build().with_max_iters((unsigned)maxit),
      gko::stop::ResidualNorm<double>::build().with_reduction_factor(1e-8))
      .with_preconditioner(mgfac).on(exec)->generate(A);
  size_t f2=free_mib(dev);
  auto logger=gko::share(gko::log::Convergence<double>::create());
  solver->add_logger(logger);
  solver->apply(b,x); exec->synchronize();
  size_t f3=free_mib(dev);
  printf("mode=%s N=%d rows=%d | iters=%ld | VRAM: matrix+vec=%zuMiB precond_gen=%zuMiB total_used=%zuMiB (free %zu->%zu)\n",
    mode.c_str(),N,n,(long)logger->get_num_iterations(), f0-f1, f1-f2, f0-f3, f0, f3);
  return 0;
}
