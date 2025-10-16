#!/usr/bin/env python3
"""
Plot asymptotic limits from combine output ROOT files.
Reads higgsCombineTest.AsymptoticLimits.mH120.root files and creates limit plots.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import mplhep
import ROOT
import argparse
from pathlib import Path

def read_limit_file(filepath):
    """
    Read asymptotic limits from a ROOT file using ROOT.
    
    Returns:
    --------
    dict with keys: 'obs', 'exp', 'exp_m2', 'exp_m1', 'exp_p1', 'exp_p2'
    """
    try:
        print(f"Reading limit file: {filepath}")
        
        # Open the ROOT file
        root_file = ROOT.TFile.Open(str(filepath))
        if not root_file or root_file.IsZombie():
            print(f"Error: Could not open ROOT file {filepath}")
            return None
        
        # Get the limit tree
        tree = root_file.Get("limit")
        if not tree:
            print(f"Error: Could not find 'limit' tree in {filepath}")
            root_file.Close()
            return None
        
        print(f"Found tree with {tree.GetEntries()} entries")
        
        # Read the limit values and quantiles
        limit_dict = {}
        
        # Loop through the tree entries
        for i in range(tree.GetEntries()):
            tree.GetEntry(i)
            
            limit_value = tree.limit
            quantile = tree.quantileExpected
            
            print(f"Entry {i}: limit={limit_value}, quantile={quantile}")
            
            # Map quantiles to their meanings with approximate matching
            if abs(quantile - (-1.0)) < 1e-6:
                limit_dict['obs'] = limit_value
            elif abs(quantile - 0.025) < 1e-6:
                limit_dict['exp_m2'] = limit_value
            elif abs(quantile - 0.16) < 1e-6:
                limit_dict['exp_m1'] = limit_value
            elif abs(quantile - 0.5) < 1e-6:
                limit_dict['exp'] = limit_value
            elif abs(quantile - 0.84) < 1e-6:
                limit_dict['exp_p1'] = limit_value
            elif abs(quantile - 0.975) < 1e-6:
                limit_dict['exp_p2'] = limit_value
        
        root_file.Close()
        print(f"Parsed limits: {limit_dict}")
        return limit_dict
        
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def collect_limits(base_dir, years=['combined'], masses=None):
    """
    Collect all limit files from the specified directory structure.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing the results
    years : list
        List of years to process (default: ['combined'])
    masses : list
        List of masses to process (if None, auto-detect from directories)
    
    Returns:
    --------
    dict: {year: {mass: limit_dict}}
    """
    results = {}
    
    for year in years:
        results[year] = {}
        year_dir = Path(base_dir) / year
        
        if not year_dir.exists():
            print(f"Warning: Directory {year_dir} does not exist")
            continue
        
        # Auto-detect masses if not specified
        if masses is None:
            mass_dirs = [d for d in year_dir.iterdir() if d.is_dir() and d.name.startswith('m')]
            detected_masses = []
            for mass_dir in mass_dirs:
                try:
                    mass = int(mass_dir.name[1:])  # Remove 'm' prefix
                    detected_masses.append(mass)
                except ValueError:
                    continue
            masses_to_process = sorted(detected_masses)
        else:
            masses_to_process = masses
        
        for mass in masses_to_process:
            mass_dir = year_dir / f"m{mass}" / f"m{mass}_model"
            limit_file = mass_dir / "higgsCombineTest.AsymptoticLimits.mH120.root"
            
            if limit_file.exists():
                limit_data = read_limit_file(str(limit_file))
                if limit_data:
                    results[year][mass] = limit_data
                    print(f"Loaded limits for {year}, mass {mass}")
            else:
                print(f"Warning: Limit file not found: {limit_file}")
    
    return results

def plot_limits(results, output_file=None, year_to_plot='combined', title_suffix="", output_dir='./plots', mode='phi'):
    """
    Create a limit plot from the collected results.
    
    Parameters:
    -----------
    results : dict
        Results from collect_limits()
    output_file : str
        Output filename for the plot (optional)
    year_to_plot : str
        Which year to plot (default: 'combined')
    title_suffix : str
        Additional text to add to the plot title
    """
    if year_to_plot not in results or not results[year_to_plot]:
        print(f"No data found for year {year_to_plot}")
        return
    
    # Extract data for plotting
    masses = sorted(results[year_to_plot].keys())
    
    obs_limits = []
    exp_limits = []
    exp_m2_limits = []
    exp_m1_limits = []
    exp_p1_limits = []
    exp_p2_limits = []
    
    for mass in masses:
        data = results[year_to_plot][mass]
        if mode == 'phi':
            factor = 1.
            # Convert r_p limits to coupling limits by taking square root
            obs_limits.append(np.sqrt(data.get('obs', np.nan)))
            exp_limits.append(np.sqrt(data.get('exp', np.nan)))
            exp_m2_limits.append(np.sqrt(data.get('exp_m2', np.nan)))
            exp_m1_limits.append(np.sqrt(data.get('exp_m1', np.nan)))
            exp_p1_limits.append(np.sqrt(data.get('exp_p1', np.nan)))
            exp_p2_limits.append(np.sqrt(data.get('exp_p2', np.nan)))
        elif mode == 'z':
            # Placeholder: direct value, or customize as needed
            factor = 1/16. # Adjust as needed for Z-prime
            obs_limits.append(np.sqrt(factor * data.get('obs', np.nan)))
            exp_limits.append(np.sqrt(factor * data.get('exp', np.nan)))
            exp_m2_limits.append(np.sqrt(factor * data.get('exp_m2', np.nan)))
            exp_m1_limits.append(np.sqrt(factor * data.get('exp_m1', np.nan)))
            exp_p1_limits.append(np.sqrt(factor * data.get('exp_p1', np.nan)))
            exp_p2_limits.append(np.sqrt(factor * data.get('exp_p2', np.nan)))
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    # Convert to numpy arrays
    masses = np.array(masses)
    obs_limits = np.array(obs_limits)
    exp_limits = np.array(exp_limits)
    exp_m2_limits = np.array(exp_m2_limits)
    exp_m1_limits = np.array(exp_m1_limits)
    exp_p1_limits = np.array(exp_p1_limits)
    exp_p2_limits = np.array(exp_p2_limits)
    
    # Set CMS style
    mplhep.style.use("CMS")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot expected limits with uncertainty bands using specified colors
    al=1
    ax.fill_between(masses, exp_m2_limits, exp_p2_limits, 
                     alpha=al, color='#ffcc00', label='Expected 95%')
    ax.fill_between(masses, exp_m1_limits, exp_p1_limits, 
                     alpha=al, color='#00953e', label='Expected 68%')
    
    
    # Plot expected and observed limits
    ax.plot(masses, exp_limits, 'k--', linewidth=2, label='Expected')
    ax.plot(masses, obs_limits, 'k-', linewidth=2, label='Observed')
       
    def scalar_no_zero(x, pos):
        if x == 0:
            return ""
        return f"{x:g}"
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(scalar_no_zero))
    if mode == 'phi':
        axt = ax.twinx()
        axt.plot(masses, obs_limits/1.5, '-', color='none', linewidth=2)
        axt.set_ylim(0, 10/1.5)
        axt.set_ylabel(r'$g_{qA}$')  # Coupling label
        axt.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(scalar_no_zero))

    # Styling
    ax.legend(loc='upper left')
    if mode == 'phi':
        ax.set_ylabel(r'$g_{q\phi}$')
        ax.set_xlabel("$\\phi/\\mathrm{A}$ mass [GeV]")
        ax.set_ylim(0,10)
    elif mode == 'z':
        ax.set_ylabel(r'$g_{q}$')
        ax.set_xlabel("Z' mass [GeV]")  # Placeholder for Z-prime
        ax.set_ylim(0,0.2)
        
    mplhep.yscale_legend(ax, soft_fail=True)
    # Set x-axis limits to exactly min/max masses (no whitespace)
    if len(masses) > 0:
        ax.set_xlim(min(masses), 300)
    # Add CMS label
    mplhep.cms.label(lumi=138, data=True)
    
    # --- Pull calculation and distribution ---
    import scipy.stats as stats
    exp_std = (np.abs(exp_p1_limits - exp_limits) + np.abs(exp_limits - exp_m1_limits)) / 2
    pulls = (obs_limits - exp_limits) / exp_std
    pulls_clean = pulls[~np.isnan(pulls)]
    mean = np.mean(pulls_clean)
    std = np.std(pulls_clean)
    print(f"\nPull mean: {mean:.3f}, std: {std:.3f}")
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.hist(pulls_clean, bins=10, alpha=0.7, color='skyblue', edgecolor='k', density=True, 
             label=fr'Pulls ($\mu$: {mean:.2f}, $\sigma$: {std:.2f})')
    x = np.linspace(-3, 3, 100)
    ax2.plot(x, stats.norm.pdf(x), 'r--', label='Normal(0,1)')
    ax2.set_xlabel('Pull: (Observed - Expected) / 1σ')
    ax2.set_ylabel('Density')
    if len(pulls_clean) >= 8:
        k2, p = stats.normaltest(pulls_clean)
        print(f"Normality test p-value: {p:.3g} (p>0.05 means consistent with normal)")
        legend_title = f"p-value: {p:.3g}"
    else:
        print(f"Normality test skipped (need at least 8 samples, got {len(pulls_clean)})")
        legend_title = "p-value: N/A"
    ax2.legend(title=legend_title, loc='upper left')
    mplhep.cms.label(lumi=138, data=True)
    mplhep.yscale_legend(ax2)

    # Save all outputs to output_dir
    os.makedirs(output_dir, exist_ok=True)
    if output_file:
        base_name = os.path.basename(output_file)
        out_path = os.path.join(output_dir, base_name)
        # Save PNG and PDF for main plot
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches='tight')
        print(f"Plot saved to {out_path} and {os.path.splitext(out_path)[0] + '.pdf'}")
        # Save PNG and PDF for pull plot
        pull_file = os.path.splitext(base_name)[0] + "_pulls.png"
        pull_path = os.path.join(output_dir, pull_file)
        fig2.savefig(pull_path, dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.splitext(pull_path)[0] + ".pdf", bbox_inches='tight')
        print(f"Pull distribution plot saved to {pull_path} and {os.path.splitext(pull_path)[0] + '.pdf'}")
    else:
        default_file = f"limits_{year_to_plot}.png"
        out_path = os.path.join(output_dir, default_file)
        # Save PNG and PDF for main plot
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches='tight')
        print(f"Plot saved to {out_path} and {os.path.splitext(out_path)[0] + '.pdf'}")
        # Save PNG and PDF for pull plot
        pull_file = f"limits_{year_to_plot}_pulls.png"
        pull_path = os.path.join(output_dir, pull_file)
        fig2.savefig(pull_path, dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.splitext(pull_path)[0] + ".pdf", bbox_inches='tight')
        print(f"Pull distribution plot saved to {pull_path} and {os.path.splitext(pull_path)[0] + '.pdf'}")
    plt.close(fig)
    plt.close(fig2)

def print_limit_table(results, year_to_plot='combined', mode='phi'):
    """
    Print a formatted table of limit values.
    """
    if year_to_plot not in results or not results[year_to_plot]:
        print(f"No data found for year {year_to_plot}")
        return
    
    masses = sorted(results[year_to_plot].keys())
    
    print(f"\nCoupling Limit Table for {year_to_plot}")
    print("="*80)
    print(f"{'Mass':<8} {'Observed':<12} {'Expected':<12} {'Exp-2σ':<12} {'Exp-1σ':<12} {'Exp+1σ':<12} {'Exp+2σ':<12}")
    print("-"*80)
    factor = 1. if mode == 'phi' else 1/16.  # Adjust factor for Z-prime if needed
    
    for mass in masses:
        data = results[year_to_plot][mass]
        # Convert r_p limits to coupling limits by taking square root
        obs = np.sqrt(factor * data.get('obs', np.nan))
        exp = np.sqrt(factor * data.get('exp', np.nan))
        exp_m2 = np.sqrt(factor * data.get('exp_m2', np.nan))
        exp_m1 = np.sqrt(factor * data.get('exp_m1', np.nan))
        exp_p1 = np.sqrt(factor *data.get('exp_p1', np.nan))
        exp_p2 = np.sqrt(factor *data.get('exp_p2', np.nan))
        
        print(f"{mass:<8} {obs:<12.4f} {exp:<12.4f} {exp_m2:<12.4f} {exp_m1:<12.4f} {exp_p1:<12.4f} {exp_p2:<12.4f}")


# Print the limit values as a JSON object
def print_limit_json(results, year_to_plot='combined', filename=None, mode='phi'):
    """
    Print the limit values as a JSON object, or save to a file if filename is given.
    """
    import json
    if year_to_plot not in results or not results[year_to_plot]:
        out = {"error": f"No data found for year {year_to_plot}"}
    else:
        masses = sorted(results[year_to_plot].keys())
        out = {}
        factor = 1. if mode == 'phi' else 1/16.  # Adjust factor for Z-prime if needed
        for mass in masses:
            data = results[year_to_plot][mass]
            out[mass] = {
                "Observed": float(np.sqrt(factor * data.get('obs', np.nan))),
                "Expected": float(np.sqrt(factor * data.get('exp', np.nan))),
                "Exp-2sigma": float(np.sqrt(factor * data.get('exp_m2', np.nan))),
                "Exp-1sigma": float(np.sqrt(factor * data.get('exp_m1', np.nan))),
                "Exp+1sigma": float(np.sqrt(factor * data.get('exp_p1', np.nan))),
                "Exp+2sigma": float(np.sqrt(factor * data.get('exp_p2', np.nan)))
            }
    json_str = json.dumps(out, indent=2)
    if filename and filename is not True:
        with open(filename, 'w') as f:
            f.write(json_str + '\n')
        print(f"JSON written to {filename}")
    else:
        print(json_str)

def main():
    parser = argparse.ArgumentParser(description='Plot asymptotic limits from combine output')
    parser.add_argument('--mode', type=str, choices=['phi', 'z'], default='phi',
                       help='Plotting mode: phi (default) or z. Use z for Z-prime option (default: phi)')
    parser.add_argument('--input', type=str, 
                       default='/home/anovak/work/zprimeqq/results_recovery_jul17',
                       help='Top-level input directory structure (default: /home/anovak/work/zprimeqq/results_recovery_jul17)')
    parser.add_argument('--years', type=str, default='combined',
                       help='Comma-separated list of years to process (default: combined)')
    parser.add_argument('--masses', type=str, default=None,
                       help='Comma-separated list of masses (default: auto-detect)')
    parser.add_argument('--output', type=str, default='./plots',
                       help='Directory for output plots (default: ./plots)')
    parser.add_argument('--year_to_plot', type=str, default='combined',
                       help='Which year to plot (default: combined)')
    parser.add_argument('--title_suffix', type=str, default='',
                       help='Additional text to add to plot title')
    parser.add_argument('--table', action='store_true',
                       help='Print a formatted table of limit values')
    parser.add_argument('--json', nargs='?', const=True, default=False,
                       help='Print a json of limit values, or save to file if a filename is given')
    # Removed --decorr_scale_wz option
    
    args = parser.parse_args()
    
    input_path = args.input

    # Parse arguments
    years = [year.strip() for year in args.years.split(',')]

    if args.masses:
        masses = [int(mass.strip()) for mass in args.masses.split(',')]
    else:
        masses = None

    # Collect limit data
    print(f"Collecting limits from {input_path}")
    print(f"Years: {years}")
    print(f"Masses: {masses if masses else 'auto-detect'}")

    results = collect_limits(input_path, years=years, masses=masses)
    
    # Print summary
    print("\nCollected data summary:")
    for year in results:
        if results[year]:
            masses_found = sorted(results[year].keys())
            print(f"  {year}: {len(masses_found)} masses ({masses_found})")
        else:
            print(f"  {year}: No data found")
    
    # Print table if requested
    if args.table:
        print_limit_table(results, year_to_plot=args.year_to_plot, mode=args.mode)

    if args.json:
        if isinstance(args.json, str):
            print_limit_json(results, year_to_plot=args.year_to_plot, filename=args.json, mode=args.mode)
        else:
            print_limit_json(results, year_to_plot=args.year_to_plot, mode=args.mode)
    
    # Create the plot
    if results[args.year_to_plot]:
        plot_limits(results, output_file=None, year_to_plot=args.year_to_plot, 
                   title_suffix=args.title_suffix, output_dir=args.output, mode=args.mode)
    else:
        print(f"No data available for plotting year {args.year_to_plot}")

if __name__ == "__main__":
    main()
