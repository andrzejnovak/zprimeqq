# Vibe-coded with Claude
import os
import argparse
import shutil
import time
from remake_utils import (
    execute_commands, 
    print_execution_summary, 
    print_fit_summary, 
    print_limit_summary,
    format_elapsed_time
)

years = {
    "2016APV": f" --year 2016APV --ipt 2,0 --irho 2,0 --iptMC 0,2 --irhoMC 1,3 ",
    "2016": f" --year 2016 --ipt 2,2 --irho 2,2 --iptMC 0,2 --irhoMC 1,3 ",
    "2017": f" --year 2017 --ipt 2,1 --irho 3,0 --iptMC 0,2 --irhoMC 1,4 ",
    "2018": f" --year 2018 --ipt 1,0 --irho 1,0 --iptMC 2,2 --irhoMC 3,4 ",
}

def read_all_masses(filename="all_masses.txt"):
    """Read all masses from a text file"""
    try:
        with open(filename, 'r') as f:
            masses = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Handle comma-separated values or space-separated values
                    if ',' in line:
                        masses.extend([int(x.strip()) for x in line.split(',') if x.strip().isdigit()])
                    else:
                        masses.extend([int(x.strip()) for x in line.split() if x.strip().isdigit()])
            return sorted(list(set(masses)))  # Remove duplicates and sort
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using default masses.")
        return [75, 85, 95]  # Default masses
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return [75, 85, 95]  # Default masses

def get_workspace_info(year, wdir, mass):
    """Get workspace directory and file for a given year"""
    if year == "combined":
        subdir = f"{wdir}/combined/m{mass}/m{mass}_model"
        workspace_file = "model_combined.root"
    else:
        subdir = f"{wdir}/{year}/m{mass}/m{mass}_model"
        workspace_file = "model_combined.root"
    return subdir, workspace_file

def main():
    # Start overall timer
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Z prime analysis workflow dispatcher')
    parser.add_argument('--make', action='store_true', help='Run make action')
    parser.add_argument('--build', action='store_true', help='Run build action')
    parser.add_argument('--fit', action='store_true', help='Run fit action')
    parser.add_argument('--limit', action='store_true', help='Run limit action')
    parser.add_argument('--combine', action='store_true', help='Combine cards from all years')
    parser.add_argument('--year', type=str, default="all",
                       help='Year(s) to process: 2016APV, 2016, 2017, 2018, all, combined, or comma-separated list (e.g., 2017,2018)')
    parser.add_argument('--mass', type=str, default="75", 
                       help='Signal mass(es): single mass (75), comma-separated list (85,95), or "all" to use all_masses.txt (default: 75)')
    parser.add_argument('-p', '--parallel', action='store_true', default=True,
                       help='Run commands in parallel (default: True)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show real-time output from commands')
    parser.add_argument('--decorr_scale_wz', action='store_true',
                       help='Add --decorr_scale_wz option to the workspace command')
    parser.add_argument('--model_type', type=str, choices=['z', 'phi'], default='phi',
                       help="Model type: 'z' or 'phi' (affects POI and workspace mapping, default: phi)")
    # Add build_dir argument
    parser.add_argument('--build_dir', type=str, default=None,
                       help='Directory for build outputs (default: results_recovery_jul17[_decorr_scale_wz])')
    args = parser.parse_args()

    # Check if at least one action is specified
    if not any([args.make, args.build, args.fit, args.limit, args.combine]):
        parser.error('At least one action must be specified: --make, --build, --fit, --limit, or --combine')

    # Parse years argument
    if args.year == "all":
        selected_years = list(years.keys())
    elif args.year == "combined":
        selected_years = ["combined"]
    else:
        selected_years = [year.strip() for year in args.year.split(',')]
        
        # Check for invalid mixing of special keywords with actual years
        if "all" in selected_years and len(selected_years) > 1:
            print("Error: 'all' cannot be combined with other year specifications")
            return
        if "combined" in selected_years and len(selected_years) > 1:
            print("Error: 'combined' cannot be combined with other year specifications")
            return
            
        # Validate years (allow "combined" as a special year)
        for year in selected_years:
            if year not in years and year != "combined":
                print(f"Error: Invalid year '{year}'. Valid years: {list(years.keys()) + ['combined']}")
                return  

    # Parse mass argument
    if args.mass == "all":
        selected_masses = read_all_masses()
        print(f"Using masses from all_masses.txt: {selected_masses}")
    else:
        try:
            selected_masses = [int(mass.strip()) for mass in args.mass.split(',')]
        except ValueError:
            print(f"Error: Invalid mass format '{args.mass}'. Use single mass (75), comma-separated list (85,95), or 'all'")
            return
    
    print(f"Selected masses: {selected_masses}")

    # Set workspace directory based on build_dir or decorr_scale_wz option
    if args.build_dir:
        wdir = args.build_dir
        if args.decorr_scale_wz:
            base_cmd = f"python rhalphalib_zprime_redo.py --opath {wdir} --tagger pnmd2prong --MCTF --tworeg  --collapse  --shift_sf_err 1.0 --muonCR --do_systematics  --do_systematics --decorr_scale_wz --force"
        else:
            base_cmd = f"python rhalphalib_zprime_redo.py --opath {wdir} --tagger pnmd2prong --MCTF --tworeg  --collapse  --shift_sf_err 1.0 --muonCR --do_systematics  --do_systematics --force"
    else:
        if args.decorr_scale_wz:
            wdir = "results_recovery_jul17_decorr_scale_wz"
            base_cmd = f"python rhalphalib_zprime_redo.py --opath {wdir} --tagger pnmd2prong --MCTF --tworeg  --collapse  --shift_sf_err 1.0 --muonCR --do_systematics  --do_systematics --decorr_scale_wz --force"
        else:
            wdir = "results_recovery_jul17"
            base_cmd = f"python rhalphalib_zprime_redo.py --opath {wdir} --tagger pnmd2prong --MCTF --tworeg  --collapse  --shift_sf_err 1.0 --muonCR --do_systematics  --do_systematics --force"
    
    if args.make:
        print(f"Make action for year(s): {args.year}, masses: {selected_masses}")
        print(f"Parallel execution     : {'ON' if args.parallel else 'OFF'}")
        print(f"Verbose output         : {'ON' if args.verbose else 'OFF'}")
        print(f"Decorr scale WZ        : {'ON' if args.decorr_scale_wz else 'OFF'}")
        print(f"Output directory       : {os.path.abspath(wdir)}")
        print()
        
        # Prepare commands for execution
        commands = []
        for year in selected_years:
            for mass in selected_masses:
                cmd_mass = f" --sigmass {mass}"
                cmd = base_cmd + cmd_mass + years[year]
                commands.append((f"{year}_m{mass}", cmd, args.verbose))
        
        print(f"Total commands to execute: {len(commands)}")
        
        # Execute commands
        results = execute_commands(commands, parallel=args.parallel, use_spinner=False)
        
        # Print summary (always show unless verbose mode captured no output)
        if not args.verbose or not args.parallel:
            print_execution_summary(results)
    
    if args.model_type == 'phi':
        t2w_line = "text2workspace.py -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel  --PO verbose --PO 'map=.*/*m{mass}:r_q[0,-15,15]'  --PO 'map=.*/*b{mass}:r_b[0,-15,15]' --PO 'map=.*/*p{mass}:r_p[0,-15,15]' model_combined.txt"
        rdf_poi = "--redefineSignalPOIs r_p"
        set_pois = "--setParameters r_p=0,r_b=0,r_q=0 --freezeParameters r_b,r_q"
    else:  # 'z prime model
        t2w_line = "text2workspace.py -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel  --PO verbose --PO 'map=.*/*[mb]{mass}:r[0,-15,15]' --PO 'map=.*/*p{mass}:r_p[0,-15,15]' model_combined.txt"
        rdf_poi = "--redefineSignalPOIs r"
        set_pois = "--setParameters r=0,r_p=0 --freezeParameters r_p"
        # t2w_line = f"text2workspace.py -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel  --PO verbose --PO 'map=.*/*m{mass}:r[1,-5,5]'  --PO 'map=.*/*b{mass}:r[1,-5,5]' --PO 'map=.*/*p{mass}:r_p[0,-15,15]' model_combined.txt"

    if args.build:
        print(f"Build action for year(s): {selected_years}, masses: {selected_masses}")
        print(f"Parallel execution      : {'ON' if args.parallel else 'OFF'}")
        print(f"Verbose output          : {'ON' if args.verbose else 'OFF'}")
        print(f"Decorr scale WZ         : {'ON' if args.decorr_scale_wz else 'OFF'}")
        print(f"Output directory        : {os.path.abspath(wdir)}")
        print()
        
        # Prepare build commands for execution
        build_commands = []
        for year in selected_years:
            for mass in selected_masses:
                if year == "combined":
                    # Build the combined workspace
                    subdir = f"{wdir}/combined/m{mass}/m{mass}_model"
                    build_cmd = f"cd {subdir} && bash build.sh"
                else:
                    # Build individual year workspace - modify text2workspace command
                    subdir = f"{wdir}/{year}/m{mass}/m{mass}_model"
                    # First update the build.sh file to use the new text2workspace parameters
                    build_script_path = f"{subdir}/build.sh"
                    try:
                        with open(build_script_path, 'r') as f:
                            build_content = f.read()
                        
                        # Replace the text2workspace.py line with our new parameters
                        lines = build_content.split('\n')
                        updated_lines = []
                        for line in lines:
                            if 'text2workspace.py' in line:
                                 # Replace with new parameters
                                updated_line = t2w_line.format(mass=mass)
                                updated_lines.append(updated_line)
                                print(f"Updated text2workspace command in {build_script_path}")
                            else:
                                updated_lines.append(line)
                        
                        # Write back the updated build script
                        with open(build_script_path, 'w') as f:
                            f.write('\n'.join(updated_lines))
                        
                    except Exception as e:
                        print(f"Warning: Could not update build script {build_script_path}: {e}")
                    
                    build_cmd = f"cd {subdir} && bash build.sh"
                build_commands.append((f"{year}_m{mass}", build_cmd, args.verbose))
        
        print(f"Total build commands to execute: {len(build_commands)}")
        
        # Execute build commands
        build_results = execute_commands(build_commands, parallel=args.parallel, use_spinner=False)
        
        # Print build summary
        if not args.verbose or not args.parallel:
            print_execution_summary(build_results, "Build Summary")
    
    if args.combine:
        print(f"Combine action for year(s): {selected_years}, masses: {selected_masses}")
        print(f"Decorr scale WZ           : {'ON' if args.decorr_scale_wz else 'OFF'}")
        print(f"Output directory          : {os.path.abspath(wdir)}")
        print()
        
        # Process each mass separately
        for mass in selected_masses:
            print(f"\n--- Processing mass {mass} ---")
            
            # Create combined directory
            combined_dir = f"{wdir}/combined/m{mass}/m{mass}_model"
            os.makedirs(combined_dir, exist_ok=True)
            
            # Generate combined build.sh script
            combine_script_path = f"{combined_dir}/build.sh"
            
            # Collect all card names from one of the years (they should be the same across years)
            sample_year = selected_years[0]
            sample_dir = f"{wdir}/{sample_year}/m{mass}/m{mass}_model"
            
            # Read the build.sh from sample year to extract card names
            try:
                with open(f"{sample_dir}/build.sh", 'r') as f:
                    sample_build = f.read()
                    
                # Extract card names from combineCards.py command
                combine_line = [line for line in sample_build.split('\n') if 'combineCards.py' in line][0]
                card_assignments = combine_line.split('combineCards.py')[1].split('>')[0].strip().split()
                
                # Copy all card files into the combined directory and generate new command
                combined_cards = []
                root_files_copied = set()  # Track copied ROOT files to avoid duplicates
                
                for assignment in card_assignments:
                    if '=' in assignment:
                        card_name, txt_file = assignment.split('=')
                        for year in selected_years:
                            if year == "combined":
                                continue  # Skip if somehow "combined" is in selected_years
                            
                            # Source file paths
                            source_dir = f"{wdir}/{year}/m{mass}/m{mass}_model"
                            source_card_path = f"{source_dir}/{txt_file}"
                            
                            # Destination file name with year suffix
                            dest_filename = f"{card_name}_{year}.txt"
                            dest_card_path = f"{combined_dir}/{dest_filename}"
                            
                            # Copy and modify the card file
                            try:
                                with open(source_card_path, 'r') as f:
                                    card_content = f.read()
                                
                                # Find ROOT file references and copy them
                                original_root_file = f"m{mass}_model.root"
                                new_root_file = f"m{mass}_model_{year}.root"
                                
                                # Copy ROOT file if not already copied
                                if new_root_file not in root_files_copied:
                                    source_root_path = f"{source_dir}/{original_root_file}"
                                    dest_root_path = f"{combined_dir}/{new_root_file}"
                                    try:
                                        shutil.copy2(source_root_path, dest_root_path)
                                        print(f"Copied {source_root_path} -> {dest_root_path}")
                                        root_files_copied.add(new_root_file)
                                    except FileNotFoundError:
                                        print(f"Warning: ROOT file not found: {source_root_path}")
                                    except Exception as e:
                                        print(f"Error copying ROOT file {source_root_path}: {e}")
                                
                                # Update card content to reference the new ROOT file
                                updated_card_content = card_content.replace(original_root_file, new_root_file)
                                
                                # Write the updated card file
                                with open(dest_card_path, 'w') as f:
                                    f.write(updated_card_content)
                                
                                print(f"Copied and updated {source_card_path} -> {dest_card_path}")
                                
                            except FileNotFoundError:
                                print(f"Warning: Card file not found: {source_card_path}")
                                continue
                            except Exception as e:
                                print(f"Error processing card {source_card_path}: {e}")
                                continue
                            
                            # Add to combined cards list (using just the filename)
                            combined_cards.append(f"{card_name}_{year}={dest_filename}")
                
                # Create the combined build.sh
                with open(combine_script_path, 'w') as f:
                    f.write(f"# Combined cards from all years for mass {mass}\n")
                    f.write("# All card files have been copied to this directory\n")
                    f.write("combineCards.py ")
                    f.write(" ".join(combined_cards))
                    f.write(" > model_combined.txt\n")
                    f.write(f"{t2w_line.format(mass=mass)}\n")
                
                # Make the script executable
                os.chmod(combine_script_path, 0o755)
                
                print(f"Created combined build script: {combine_script_path}")
                print(f"Combined directory: {combined_dir}")
                
            except Exception as e:
                print(f"Error creating combined build script for mass {mass}: {e}")
                continue
    
    fit_cmd_opts = f" --cminDefaultMinimizerStrategy 0 --cminFallbackAlgo Minuit2,0:0.4 {rdf_poi} {set_pois}"
    if args.fit:
        print(f"Fit action for year(s): {selected_years}, masses: {selected_masses}")
        print(f"Parallel execution     : {'ON' if args.parallel else 'OFF'}")
        print(f"Verbose output         : {'ON' if args.verbose else 'OFF'}")
        print(f"Decorr scale WZ        : {'ON' if args.decorr_scale_wz else 'OFF'}")
        print(f"Output directory       : {os.path.abspath(wdir)}")
        print()
        
        # Prepare fit commands for execution
        fit_commands = []
        for year in selected_years:
            for mass in selected_masses:
                subdir, workspace_file = get_workspace_info(year, wdir, mass)
                
                # Create the combine command
                combine_cmd = f"combine -M FitDiagnostics {fit_cmd_opts} --saveShapes --saveWithUncertainties {workspace_file}"
                
                # Save the command to a text file in the workspace directory
                cmd_file_path = f"{subdir}/fit_command.txt"
                os.makedirs(subdir, exist_ok=True)  # Ensure directory exists for command file
                with open(cmd_file_path, 'w') as f:
                    f.write(f"# Fit command for {year}, mass {mass}\n")
                    f.write(f"# Generated on {os.popen('date').read().strip()}\n")
                    f.write(f"# Working directory: {subdir}\n\n")
                    f.write(combine_cmd + "\n")
                
                # Full command with cd and logging
                log_file = f"fit_log.txt"  # Use relative path since we're cd'ing into the directory
                fit_cmd = f"cd {subdir} && {combine_cmd} 2>&1 | tee {log_file}"
                fit_commands.append((f"{year}_m{mass}", fit_cmd, args.verbose))
                
                print(f"Saved fit command to   : {os.path.abspath(cmd_file_path)}")
                print(f"Fit log will be saved to: {os.path.abspath(subdir)}/{log_file}")
        
        print(f"Total fit commands to execute: {len(fit_commands)}")
        
        # Execute fit commands
        fit_results = execute_commands(fit_commands, parallel=args.parallel, use_spinner=True)
        
        # Print fit summary
        if not args.verbose or not args.parallel:
            print_fit_summary(fit_results)

    
    if args.limit:
        print(f"Limit action for year(s): {selected_years}, masses: {selected_masses}")
        print(f"Parallel execution      : {'ON' if args.parallel else 'OFF'}")
        print(f"Verbose output          : {'ON' if args.verbose else 'OFF'}")
        print(f"Decorr scale WZ         : {'ON' if args.decorr_scale_wz else 'OFF'}")
        print(f"Output directory        : {os.path.abspath(wdir)}")
        print()
        
        # Limit-specific command options (can be modified independently from fit)
        limit_cmd_opts = f" --cminDefaultMinimizerStrategy 0 --cminFallbackAlgo Minuit2,0:0.4 {rdf_poi} {set_pois}"
        # Prepare limit commands for execution
        limit_commands = []
        for year in selected_years:
            for mass in selected_masses:
                subdir, workspace_file = get_workspace_info(year, wdir, mass)
                
                # Create the combine command
                combine_cmd = f"combine -M AsymptoticLimits {limit_cmd_opts} {workspace_file}"
                
                # Save the command to a text file in the workspace directory
                cmd_file_path = f"{subdir}/limit_command.txt"
                os.makedirs(subdir, exist_ok=True)  # Ensure directory exists for command file
                with open(cmd_file_path, 'w') as f:
                    f.write(f"# Limit command for {year}, mass {mass}\n")
                    f.write(f"# Generated on {os.popen('date').read().strip()}\n")
                    f.write(f"# Working directory: {subdir}\n\n")
                    f.write(combine_cmd + "\n")
                
                # Full command with cd and logging
                log_file = f"limit_log.txt"  # Use relative path since we're cd'ing into the directory
                limit_cmd = f"cd {subdir} && {combine_cmd} 2>&1 | tee {log_file}"
                limit_commands.append((f"{year}_m{mass}", limit_cmd, args.verbose))
                
                print(f"Saved limit command to  : {os.path.abspath(cmd_file_path)}")
                print(f"Limit log will be saved to: {os.path.abspath(subdir)}/{log_file}")
        
        print(f"Total limit commands to execute: {len(limit_commands)}")
        
        # Execute limit commands
        limit_results = execute_commands(limit_commands, parallel=args.parallel, use_spinner=True)
        
        # Print limit summary
        if not args.verbose or not args.parallel:
            print_limit_summary(limit_results)

    # Print overall timing summary
    end_time = time.time()
    total_elapsed = end_time - start_time
    
    print(f"\n{'='*50}")
    print(f"OVERALL EXECUTION SUMMARY")
    print(f"{'='*50}")
    print(f"Total execution time: {format_elapsed_time(total_elapsed)}")
    
    # Count actions performed
    actions_performed = []
    if args.make:
        actions_performed.append("make")
    if args.build:
        actions_performed.append("build")
    if args.combine:
        actions_performed.append("combine")
    if args.fit:
        actions_performed.append("fit")
    if args.limit:
        actions_performed.append("limit")
    
    print(f"Actions performed: {', '.join(actions_performed)}")
    print(f"Years processed: {', '.join(selected_years)}")
    print(f"Masses processed: {', '.join(map(str, selected_masses))}")
    print(f"Total mass/year combinations: {len(selected_years) * len(selected_masses)}")
    print(f"Parallel execution: {'ON' if args.parallel else 'OFF'}")
    print(f"Decorr scale WZ: {'ON' if args.decorr_scale_wz else 'OFF'}")
    print(f"Output directory: {os.path.abspath(wdir)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
