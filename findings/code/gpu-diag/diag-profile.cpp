// Per-operation timing breakdown of the MG-preconditioned CG solve, via
// Ginkgo's ProfilerHook nested summary (GPU timer). Shows where time goes
// inside an iteration: SpMV, MG smoother/restrict/coarse/prolong, dot-products,
// axpy, copies, allocations. mode=double|single. N -> N*N rows (per-rank shard).
#include <ginkgo/ginkgo.hpp>
#include <cstdio>
#include <string>
#include <iostream>
using it=int;

template<typename V>
void solve(std::shared_ptr<gko::DpcppExecutor> exec, int N, int maxit, const char* tag){
  using mtx=gko::matrix::Csr<V,it>; using dn=gko::matrix::Dense<V>;
  using cg=gko::solver::Cg<V>; using mg=gko::solver::Multigrid;
  using pgm=gko::multigrid::Pgm<V,it>; using ir=gko::solver::Ir<V>;
  using bj=gko::preconditioner::Jacobi<V,it>; using Iter=gko::stop::Iteration;
  int n=N*N;
  gko::matrix_data<V,it> md(gko::dim<2>{(gko::dim<2>::dimension_type)n,(gko::dim<2>::dimension_type)n});
  md.nonzeros.reserve(5*n);
  for(int j=0;j<N;++j)for(int i=0;i<N;++i){int r=j*N+i;
    md.nonzeros.push_back({r,r,(V)4}); if(i>0)md.nonzeros.push_back({r,r-1,(V)-1}); if(i<N-1)md.nonzeros.push_back({r,r+1,(V)-1});
    if(j>0)md.nonzeros.push_back({r,r-N,(V)-1}); if(j<N-1)md.nonzeros.push_back({r,r+N,(V)-1});}
  auto A=gko::share(mtx::create(exec)); A->read(md);
  auto b=gko::share(dn::create(exec,gko::dim<2>{(gko::dim<2>::dimension_type)n,1})); b->fill((V)1);
  auto x=gko::share(dn::create(exec,gko::dim<2>{(gko::dim<2>::dimension_type)n,1})); x->fill((V)0);
  exec->synchronize();

  auto sm=gko::share(ir::build().with_solver(bj::build().with_max_block_size(1u).on(exec)).with_relaxation_factor((V)0.9).with_criteria(Iter::build().with_max_iters(1u)).on(exec));
  auto co=gko::share(cg::build().with_preconditioner(bj::build().with_max_block_size(1u).on(exec)).with_criteria(Iter::build().with_max_iters(20u)).on(exec));
  auto mgfac=gko::share(mg::build().with_max_levels(20u).with_min_coarse_rows(64000u)
    .with_pre_smoother(sm).with_post_uses_pre(true)
    .with_mg_level(pgm::build().with_deterministic(false).on(exec))
    .with_coarsest_solver(co).with_criteria(Iter::build().with_max_iters(1u)).on(exec));
  auto solver=cg::build().with_criteria(Iter::build().with_max_iters((unsigned)maxit),
      gko::stop::ResidualNorm<V>::build().with_reduction_factor((V)1e-2))  // relTol 0.01 like OGL
      .with_preconditioner(mgfac).on(exec)->generate(A);

  // GPU-accurate timer + nested summary -> stdout
  auto prof=gko::log::ProfilerHook::create_nested_summary(std::make_shared<gko::CpuTimer>(),
      std::make_unique<gko::log::ProfilerHook::TableSummaryWriter>(std::cout, std::string("=== ")+tag+" nested runtime summary ==="));
  prof->set_synchronization(true);
  exec->add_logger(prof);
  solver->add_logger(prof);
  // warm-up (build hierarchy already done in generate); now timed solve
  solver->apply(b,x); exec->synchronize();
  exec->remove_logger(prof); solver->remove_logger(prof);  // triggers summary write on destruction
  prof.reset();
}

int main(int argc,char**argv){
  std::string mode=argc>1?argv[1]:"single"; int N=argc>2?std::atoi(argv[2]):1450; int maxit=argc>3?std::atoi(argv[3]):60;
  auto exec=gko::DpcppExecutor::create(0,gko::OmpExecutor::create());
  printf("# profiling mode=%s N=%d rows=%d\n",mode.c_str(),N,N*N); fflush(stdout);
  if(mode=="double") solve<double>(exec,N,maxit,"double"); else solve<float>(exec,N,maxit,"single");
  return 0;
}
