- Depends on combine, rhalphalib
- Run like 
```
python jeff_rhalpha.py --opath dummyx/ --ipt 2 --irho 2 --MCTF --irhoMC 2 --iptMC 2 --tagger pnmd2prong_0p05 --root_file TEMPLATES_blind2reg.root --year 2017 --tworeg -vv
```

Run postfit plots like 
```
combine_postfits -i fitDiagnosticsTest.root --data --style style.yml --sigs b150,m150 --project-signals 3,3 --rmap '{"m150": "r_q", "b150":"r_b"}' --bkgs top,vlep,wqq,zqq,zbb,hbb --onto 2017_qcd -p
```



# Redo

# Setup env (using pixi)

- Install the base env
    ```bash
    pixi shell
    ```
- Checkout combine (using an old version as used in the analysis)
    ```
    git -c advice.detachedHead=false clone --depth 1 --branch v9.2.1 https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
    ```
- Apply the pixi patch to the Makefile of combine (need to link to the dependencies in the pixi env)
    ```
    pixi run combine-patch
    ```
- Build it 
    ```
    pixi run combine-build
    ```
- Link combine and test it 
    ```
    bash setup/pixi-link-combine.sh HiggsAnalysis/CombinedLimit/
    combine
    ```
- You should see:
    ```
    >>> combine

    RooFit v3.60 -- Developed by Wouter Verkerke and David Kirkby
                    Copyright (C) 2000-2013 NIKHEF, University of California & Stanford University
                    All rights reserved, please read http://roofit.sourceforge.net/license.txt

    <<< Combine >>>
    <<< v9.2.1 >>>
    Missing datacard file
    Usage: combine [options]
    Use combine --help to get a list of all the allowed methods and options
    ```
- Finally verify path is correct
    ```
    which combine
    ```
- Install `rhalphabet`
    ```
    python -m pip install git+https://github.com/nsmith-/rhalphalib.git@v0.3.0
    ```

## Let's try

```
COUPLING=-
POSTFIX=9Jul24_v1
YEAR=combination

python3 run_limits.py --postfix 11July25 --year 2016APV --r_p --sigmass 50 --prefix newtest --make
python3 run_limits.py --postfix 11July25 --year 2016 --r_p --sigmass 50 --prefix newtest --make
python3 run_limits.py --postfix 11July25 --year 2017 --r_p --sigmass 50 --prefix newtest --make
python3 run_limits.py --postfix 11July25 --year 2018 --r_p --sigmass 50 --prefix newtest --make


```

## Using remake.py (Modern Workflow Dispatcher)

```bash
# Full combined fit workflow
python remake.py  --mass 75 --year all --make --combine -p 
python remake.py  --mass 75 --year combined --build -p
python remake.py  --mass 75 --year combined --fit -p

# Or step by step:
python remake.py --make --mass 75                    # Generate templates
python remake.py --combine --year all --mass 75      # Combine all years  
python remake.py --build --year combined --mass 75   # Build workspace
python remake.py --fit --year combined --mass 75     # Run fit
```
