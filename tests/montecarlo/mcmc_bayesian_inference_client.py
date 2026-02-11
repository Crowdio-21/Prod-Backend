#!/usr/bin/env python3
"""
Markov Chain Monte Carlo (MCMC) Client for Bayesian Inference

This script demonstrates:
1. Distributed MCMC simulation using Metropolis-Hastings algorithm
2. Bayesian parameter estimation for normal distribution
3. Task offloading to worker devices
4. Chain convergence diagnostics and result aggregation

Mathematical Background:
MCMC methods are used for sampling from probability distributions where direct
sampling is difficult. The Metropolis-Hastings algorithm creates a Markov chain
whose stationary distribution is the target posterior distribution. Multiple
chains can be run in parallel to improve sampling and assess convergence.

Use Case: Estimating mean (μ) and standard deviation (σ) of a normal distribution
given observed data, using Bayesian inference with uninformative priors.
"""

import asyncio
import sys
import os
import json
import time

# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from developer_sdk import connect, map as distributed_map, disconnect


def mcmc_bayesian_inference_worker(config):
    """
    Worker function to perform MCMC sampling using Metropolis-Hastings algorithm
    
    Args:
        config: Dictionary containing:
            - data: Observed data points
            - num_iterations: Number of MCMC iterations
            - burn_in: Number of initial samples to discard
            - chain_id: Identifier for this chain
    
    Returns:
        Dictionary containing MCMC samples and diagnostics
    """
    import random
    import math
    import time
    
    start = time.time()
    
    try:
        # Extract configuration
        data = config['data']
        num_iterations = config['num_iterations']
        burn_in = config.get('burn_in', 1000)
        chain_id = config.get('chain_id', 0)
        
        # Set random seed for reproducibility but unique per chain
        random.seed(chain_id * 12345)
        
        n = len(data)
        data_mean = sum(data) / n
        data_var = sum((x - data_mean) ** 2 for x in data) / n
        
        # Initialize parameters (mu, sigma)
        # Start from data statistics with some noise
        current_mu = data_mean + random.gauss(0, 1)
        current_sigma = math.sqrt(data_var) + random.uniform(0, 1)
        
        # Calculate initial log-likelihood
        def log_likelihood(mu, sigma, data):
            """Calculate log-likelihood of data given parameters"""
            if sigma <= 0:
                return -float('inf')
            
            ll = 0
            for x in data:
                ll += -0.5 * math.log(2 * math.pi * sigma**2)
                ll += -0.5 * ((x - mu) / sigma) ** 2
            return ll
        
        # Log prior (uninformative)
        def log_prior(mu, sigma):
            """Log prior probability"""
            if sigma <= 0:
                return -float('inf')
            # Uniform prior on mu, log-uniform on sigma
            return -math.log(sigma)
        
        def log_posterior(mu, sigma, data):
            """Log posterior probability"""
            return log_likelihood(mu, sigma, data) + log_prior(mu, sigma)
        
        current_log_post = log_posterior(current_mu, current_sigma, data)
        
        # Storage for samples
        mu_samples = []
        sigma_samples = []
        
        # Proposal standard deviations
        proposal_sd_mu = 0.5
        proposal_sd_sigma = 0.1
        
        accepted = 0
        
        # MCMC iterations
        for iteration in range(num_iterations + burn_in):
            # Propose new parameters
            proposed_mu = current_mu + random.gauss(0, proposal_sd_mu)
            proposed_sigma = current_sigma + random.gauss(0, proposal_sd_sigma)
            
            if proposed_sigma > 0:
                # Calculate acceptance ratio
                proposed_log_post = log_posterior(proposed_mu, proposed_sigma, data)
                log_acceptance = proposed_log_post - current_log_post
                
                # Accept or reject
                if log_acceptance > 0 or random.random() < math.exp(log_acceptance):
                    current_mu = proposed_mu
                    current_sigma = proposed_sigma
                    current_log_post = proposed_log_post
                    accepted += 1
            
            # Store samples after burn-in
            if iteration >= burn_in:
                mu_samples.append(current_mu)
                sigma_samples.append(current_sigma)
        
        acceptance_rate = accepted / (num_iterations + burn_in)
        
        # Calculate posterior statistics
        mu_mean = sum(mu_samples) / len(mu_samples)
        mu_var = sum((x - mu_mean) ** 2 for x in mu_samples) / len(mu_samples)
        mu_std = math.sqrt(mu_var)
        
        sigma_mean = sum(sigma_samples) / len(sigma_samples)
        sigma_var = sum((x - sigma_mean) ** 2 for x in sigma_samples) / len(sigma_samples)
        sigma_std = math.sqrt(sigma_var)
        
        # Calculate quantiles (95% credible interval)
        mu_sorted = sorted(mu_samples)
        sigma_sorted = sorted(sigma_samples)
        
        idx_025 = int(0.025 * len(mu_sorted))
        idx_975 = int(0.975 * len(mu_sorted))
        
        mu_ci_lower = mu_sorted[idx_025]
        mu_ci_upper = mu_sorted[idx_975]
        sigma_ci_lower = sigma_sorted[idx_025]
        sigma_ci_upper = sigma_sorted[idx_975]
        
        latency_ms = int((time.time() - start) * 1000)
        
        result = {
            "chain_id": chain_id,
            "num_iterations": num_iterations,
            "burn_in": burn_in,
            "acceptance_rate": round(acceptance_rate, 4),
            "posterior_mu": {
                "mean": round(mu_mean, 6),
                "std": round(mu_std, 6),
                "ci_95_lower": round(mu_ci_lower, 6),
                "ci_95_upper": round(mu_ci_upper, 6)
            },
            "posterior_sigma": {
                "mean": round(sigma_mean, 6),
                "std": round(sigma_std, 6),
                "ci_95_lower": round(sigma_ci_lower, 6),
                "ci_95_upper": round(sigma_ci_upper, 6)
            },
            "mu_samples": mu_samples[-100:],  # Last 100 samples for diagnostics
            "sigma_samples": sigma_samples[-100:],
            "latency_ms": latency_ms,
            "status": "success"
        }
        
        print(f"[Worker Chain {chain_id}] μ={mu_mean:.4f}±{mu_std:.4f}, σ={sigma_mean:.4f}±{sigma_std:.4f}, Accept={acceptance_rate:.2%}")
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        latency_ms = int((time.time() - start) * 1000)
        
        return {
            "chain_id": config.get('chain_id', -1),
            "latency_ms": latency_ms,
            "status": "error",
            "error": str(e)
        }


# =========================================================
# 📊 RESULT AGGREGATION & CONVERGENCE DIAGNOSTICS
# =========================================================
def aggregate_mcmc_results(results):
    """
    Aggregate results from distributed MCMC chains
    Implements Gelman-Rubin convergence diagnostic (R-hat)
    
    Args:
        results: List of worker results
    
    Returns:
        Dictionary with aggregated posterior statistics and diagnostics
    """
    parsed = []
    
    for r in results:
        if isinstance(r, dict):
            parsed.append(r)
        elif isinstance(r, str):
            try:
                parsed.append(json.loads(r))
            except:
                try:
                    import ast
                    parsed.append(ast.literal_eval(r))
                except:
                    parsed.append({"status": "error", "error": "Unparseable result"})
        else:
            parsed.append({"status": "error", "error": "Unknown result type"})
    
    valid = [r for r in parsed if r.get("status") == "success"]
    
    if not valid:
        return {
            "error": "All chains failed or could not be parsed",
            "num_chains": 0,
            "error_count": len(results)
        }
    
    num_chains = len(valid)
    
    # Calculate Gelman-Rubin statistic (R-hat) for convergence
    def calculate_rhat(chain_samples):
        """Calculate R-hat statistic for convergence"""
        if len(chain_samples) < 2:
            return None
        
        m = len(chain_samples)  # Number of chains
        n = len(chain_samples[0])  # Samples per chain
        
        # Within-chain variance
        chain_means = [sum(chain) / len(chain) for chain in chain_samples]
        chain_vars = [
            sum((x - mean) ** 2 for x in chain) / (len(chain) - 1)
            for chain, mean in zip(chain_samples, chain_means)
        ]
        W = sum(chain_vars) / m
        
        # Between-chain variance
        overall_mean = sum(chain_means) / m
        B = n * sum((mean - overall_mean) ** 2 for mean in chain_means) / (m - 1)
        
        # Pooled variance
        var_plus = ((n - 1) * W + B) / n
        
        # R-hat
        if W == 0:
            return None
        rhat = (var_plus / W) ** 0.5
        return rhat
    
    # Extract samples from all chains
    mu_chain_samples = [r['mu_samples'] for r in valid]
    sigma_chain_samples = [r['sigma_samples'] for r in valid]
    
    # Calculate R-hat
    rhat_mu = calculate_rhat(mu_chain_samples)
    rhat_sigma = calculate_rhat(sigma_chain_samples)
    
    # Pool all samples for overall posterior
    all_mu_samples = []
    all_sigma_samples = []
    for r in valid:
        all_mu_samples.extend(r['mu_samples'])
        all_sigma_samples.extend(r['sigma_samples'])
    
    # Overall posterior statistics
    mu_mean = sum(all_mu_samples) / len(all_mu_samples)
    mu_var = sum((x - mu_mean) ** 2 for x in all_mu_samples) / len(all_mu_samples)
    mu_std = mu_var ** 0.5
    
    sigma_mean = sum(all_sigma_samples) / len(all_sigma_samples)
    sigma_var = sum((x - sigma_mean) ** 2 for x in all_sigma_samples) / len(all_sigma_samples)
    sigma_std = sigma_var ** 0.5
    
    # Credible intervals
    mu_sorted = sorted(all_mu_samples)
    sigma_sorted = sorted(all_sigma_samples)
    
    idx_025 = int(0.025 * len(mu_sorted))
    idx_975 = int(0.975 * len(mu_sorted))
    
    mu_ci_lower = mu_sorted[idx_025]
    mu_ci_upper = mu_sorted[idx_975]
    sigma_ci_lower = sigma_sorted[idx_025]
    sigma_ci_upper = sigma_sorted[idx_975]
    
    # Average acceptance rate
    avg_acceptance = sum(r['acceptance_rate'] for r in valid) / num_chains
    
    # Latency statistics
    latencies = [r['latency_ms'] for r in valid]
    
    return {
        "num_chains": num_chains,
        "total_samples": len(all_mu_samples),
        "posterior_mu": {
            "mean": round(mu_mean, 6),
            "std": round(mu_std, 6),
            "ci_95_lower": round(mu_ci_lower, 6),
            "ci_95_upper": round(mu_ci_upper, 6)
        },
        "posterior_sigma": {
            "mean": round(sigma_mean, 6),
            "std": round(sigma_std, 6),
            "ci_95_lower": round(sigma_ci_lower, 6),
            "ci_95_upper": round(sigma_ci_upper, 6)
        },
        "convergence_diagnostics": {
            "rhat_mu": round(rhat_mu, 4) if rhat_mu else None,
            "rhat_sigma": round(rhat_sigma, 4) if rhat_sigma else None,
            "converged": (rhat_mu is not None and rhat_sigma is not None and 
                         rhat_mu < 1.1 and rhat_sigma < 1.1)
        },
        "avg_acceptance_rate": round(avg_acceptance, 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "error_count": len(results) - len(valid),
        "chain_results": valid
    }


# =========================================================
# 🚀 DISTRIBUTED EXECUTION
# =========================================================
async def run_distributed_mcmc(data, num_chains=4, num_iterations=10000, burn_in=2000, foreman_host="localhost"):
    """
    Run distributed MCMC for Bayesian inference
    
    Args:
        data: Observed data for inference
        num_chains: Number of independent MCMC chains
        num_iterations: Number of MCMC iterations per chain
        burn_in: Number of initial samples to discard
        foreman_host: Hostname/IP of foreman server
    """
    print("\n" + "=" * 70)
    print("🔗 DISTRIBUTED MARKOV CHAIN MONTE CARLO (MCMC) - BAYESIAN INFERENCE")
    print("=" * 70)
    
    print(f"\n📡 Connecting to foreman at {foreman_host}:9000...")
    await connect(foreman_host, 9000)
    print("✅ Connected")
    
    # Data statistics
    data_mean = sum(data) / len(data)
    data_std = (sum((x - data_mean) ** 2 for x in data) / len(data)) ** 0.5
    
    print(f"\n📊 Observed Data:")
    print(f"   Sample size: {len(data)}")
    print(f"   Sample mean: {data_mean:.4f}")
    print(f"   Sample std:  {data_std:.4f}")
    
    print(f"\n⛓️  MCMC Configuration:")
    print(f"   Number of chains:     {num_chains}")
    print(f"   Iterations per chain: {num_iterations:,}")
    print(f"   Burn-in period:       {burn_in:,}")
    print(f"   Total samples:        {num_chains * num_iterations:,}")
    
    # Create configurations for each chain
    configs = []
    for chain_id in range(num_chains):
        configs.append({
            'data': data,
            'num_iterations': num_iterations,
            'burn_in': burn_in,
            'chain_id': chain_id
        })
    
    start_time = time.time()
    
    print("\n⏳ Dispatching MCMC chains to workers...")
    results = await distributed_map(mcmc_bayesian_inference_worker, configs)
    
    execution_time = time.time() - start_time
    
    aggregated = aggregate_mcmc_results(results)
    
    print("\n" + "-" * 70)
    print("📊 BAYESIAN INFERENCE RESULTS")
    print("-" * 70)
    
    if "error" not in aggregated or aggregated["num_chains"] > 0:
        print(f"\n🎯 Posterior Distribution for μ (mean):")
        print(f"   Estimate:       {aggregated['posterior_mu']['mean']:.6f}")
        print(f"   Std. Error:     {aggregated['posterior_mu']['std']:.6f}")
        print(f"   95% CI:         [{aggregated['posterior_mu']['ci_95_lower']:.6f}, {aggregated['posterior_mu']['ci_95_upper']:.6f}]")
        print(f"   True value:     {data_mean:.6f}")
        
        print(f"\n🎯 Posterior Distribution for σ (std dev):")
        print(f"   Estimate:       {aggregated['posterior_sigma']['mean']:.6f}")
        print(f"   Std. Error:     {aggregated['posterior_sigma']['std']:.6f}")
        print(f"   95% CI:         [{aggregated['posterior_sigma']['ci_95_lower']:.6f}, {aggregated['posterior_sigma']['ci_95_upper']:.6f}]")
        print(f"   True value:     {data_std:.6f}")
        
        print(f"\n📈 Convergence Diagnostics (Gelman-Rubin R̂):")
        conv = aggregated['convergence_diagnostics']
        if conv['rhat_mu'] is not None:
            mu_status = "✅ Converged" if conv['rhat_mu'] < 1.1 else "⚠️  Not converged"
            sigma_status = "✅ Converged" if conv['rhat_sigma'] < 1.1 else "⚠️  Not converged"
            print(f"   R̂ for μ:        {conv['rhat_mu']:.4f} {mu_status}")
            print(f"   R̂ for σ:        {conv['rhat_sigma']:.4f} {sigma_status}")
            print(f"   Overall:        {'✅ All chains converged' if conv['converged'] else '⚠️  Increase iterations'}")
        
        print(f"\n⚙️  MCMC Statistics:")
        print(f"   Chains completed:     {aggregated['num_chains']}")
        print(f"   Total samples:        {aggregated['total_samples']:,}")
        print(f"   Avg acceptance rate:  {aggregated['avg_acceptance_rate']:.2%}")
        
        print(f"\n⚡ Performance:")
        print(f"   Total execution time: {execution_time:.2f} seconds")
        print(f"   Avg chain latency:    {aggregated['avg_latency_ms']:.1f} ms")
        print(f"   Throughput:           {aggregated['total_samples'] / execution_time:,.0f} samples/sec")
        
        if aggregated.get("error_count", 0) > 0:
            print(f"\n⚠️  Failed chains: {aggregated['error_count']}")
        
        print("\n📋 Chain-level Results:")
        for r in aggregated.get("chain_results", []):
            print(f"Chain {r['chain_id']}: "
                  f"μ={r['posterior_mu']['mean']:.4f}±{r['posterior_mu']['std']:.4f}, "
                  f"σ={r['posterior_sigma']['mean']:.4f}±{r['posterior_sigma']['std']:.4f}, "
                  f"Accept={r['acceptance_rate']:.2%}")
    else:
        print(f"\n❌ Error: {aggregated['error']}")
    
    await disconnect()
    
    return aggregated


# =========================================================
# 🏁 MAIN
# =========================================================
async def main():
    """
    Main entry point for MCMC Bayesian inference
    """
    # Generate synthetic observed data
    # True parameters: μ = 5.0, σ = 2.0
    import random
    random.seed(42)
    
    true_mu = 5.0
    true_sigma = 2.0
    sample_size = 100
    
    # Generate data from normal distribution
    observed_data = [random.gauss(true_mu, true_sigma) for _ in range(sample_size)]
    
    # Parse command line arguments
    num_chains = 4
    num_iterations = 10000
    burn_in = 2000
    foreman_host = "localhost"
    
    if len(sys.argv) > 1:
        try:
            num_chains = int(sys.argv[1])
        except ValueError:
            foreman_host = sys.argv[1]
    
    if len(sys.argv) > 2:
        try:
            num_iterations = int(sys.argv[2])
        except ValueError:
            pass
    
    if len(sys.argv) > 3:
        foreman_host = sys.argv[3]
    
    print(f"\n📝 Simulation Configuration:")
    print(f"   True μ: {true_mu}")
    print(f"   True σ: {true_sigma}")
    print(f"   Sample size: {sample_size}")
    print(f"   Chains: {num_chains}")
    print(f"   Iterations: {num_iterations:,}")
    print(f"   Foreman: {foreman_host}")
    
    await run_distributed_mcmc(observed_data, num_chains, num_iterations, burn_in, foreman_host)


if __name__ == "__main__":
    asyncio.run(main())
